import argparse
from typing import Optional, Tuple

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from indexer_topk_reducesum import indexer_topk_reducesum_interface
from indexer_bwd import indexer_bwd_interface
from sparse_mla_fwd import sparse_mla_fwd_interface
from sparse_mla_bwd import sparse_mla_bwd
from sparse_mla_topk_reducesum import sparse_mla_topk_reducesum_interface


def _dtype_from_arg(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _build_offsets(seq_len: int, splits: int) -> torch.Tensor:
    if splits <= 1:
        return torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    step = seq_len // splits
    offsets = [0]
    for i in range(1, splits):
        offsets.append(i * step)
    offsets.append(seq_len)
    return torch.tensor(offsets, dtype=torch.int32, device="cuda")


def _make_inputs(
    seq_len: int,
    heads: int,
    dim: int,
    tail_dim: int,
    index_dim: int,
    dtype: torch.dtype,
    splits: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn((seq_len, heads, dim + tail_dim), device="cuda", dtype=dtype)
    kv = torch.randn((seq_len, dim + tail_dim), device="cuda", dtype=dtype)
    index_q = torch.randn((seq_len, heads, index_dim), device="cuda", dtype=dtype)
    weights = torch.randn((seq_len, heads), device="cuda", dtype=dtype)
    index_k = torch.randn((seq_len, index_dim), device="cuda", dtype=dtype)
    offsets = _build_offsets(seq_len, splits)
    return q, kv, index_q, index_k, weights, offsets


def _run_dsa_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    dim_v: int,
    sm_scale: Optional[float],
):
    topk_indices, index_score = indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)
    o, lse = sparse_mla_fwd_interface(
        q,
        kv.unsqueeze(-2),
        topk_indices.unsqueeze(-2),
        offsets,
        sm_scale=sm_scale,
        d_v=dim_v,
    )
    return o, lse, topk_indices, index_score


def _run_dsa_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    offsets: torch.Tensor,
    topk_indices: torch.Tensor,
    index_score: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    dim_v: int,
    sm_scale: Optional[float],
):
    attn_score = sparse_mla_topk_reducesum_interface(
        q,
        kv.unsqueeze(-2),
        topk_indices.unsqueeze(-2),
        lse,
        offsets,
        dim_v=dim_v,
    ).squeeze(-2)
    sparse_mla_bwd(
        q,
        kv.unsqueeze(-2),
        o,
        do,
        topk_indices.unsqueeze(-2),
        lse,
        offsets,
        sm_scale=sm_scale,
    )
    indexer_bwd_interface(index_q, weights, index_k, attn_score, index_score, topk_indices, offsets)


def _time_cuda_events(fn, iters: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _get_profiler_ms(prof, key: str) -> float:
    for item in prof.key_averages():
        if item.key == key:
            return item.self_cuda_time_total / 1000.0
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--tail-dim", type=int, default=64)
    parser.add_argument("--index-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sm-scale", type=float, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dtype = _dtype_from_arg(args.dtype)
    if args.dim != 512 or args.tail_dim != 64:
        raise ValueError("dim must be 512 and tail-dim must be 64 for sparse_mla kernels.")
    if args.topk & (args.topk - 1) != 0:
        raise ValueError("topk must be a power of 2.")
    if args.heads > 64 or args.heads % 8 != 0:
        raise ValueError("heads must be <= 64 and divisible by 8.")
    if args.seq_len < args.topk:
        raise ValueError("seq-len must be >= topk.")
    if args.splits > 1:
        step = args.seq_len // args.splits
        last = args.seq_len - step * (args.splits - 1)
        if step < args.topk or last < args.topk:
            raise ValueError("each split must be >= topk.")

    q, kv, index_q, index_k, weights, offsets = _make_inputs(
        args.seq_len,
        args.heads,
        args.dim,
        args.tail_dim,
        args.index_dim,
        dtype,
        args.splits,
        args.seed,
    )
    do = torch.randn((args.seq_len, args.heads, args.dim), device="cuda", dtype=dtype)

    with torch.no_grad():
        for _ in range(args.warmup):
            o, lse, topk_indices, index_score = _run_dsa_forward(
                q, kv, index_q, index_k, weights, offsets, args.topk, args.dim, args.sm_scale
            )
            _run_dsa_backward(
                q,
                kv,
                index_q,
                index_k,
                weights,
                offsets,
                topk_indices,
                index_score,
                o,
                lse,
                do,
                args.dim,
                args.sm_scale,
            )
        torch.cuda.synchronize()

    o_base, lse_base, topk_indices_base, index_score_base = _run_dsa_forward(
        q, kv, index_q, index_k, weights, offsets, args.topk, args.dim, args.sm_scale
    )
    attn_score_base = sparse_mla_topk_reducesum_interface(
        q,
        kv.unsqueeze(-2),
        topk_indices_base.unsqueeze(-2),
        lse_base,
        offsets,
        dim_v=args.dim,
    ).squeeze(-2)

    def run_indexer_fwd():
        indexer_topk_reducesum_interface(index_q, weights, index_k, args.topk, offsets)

    def run_dsa_fwd():
        _run_dsa_forward(q, kv, index_q, index_k, weights, offsets, args.topk, args.dim, args.sm_scale)

    def run_indexer_bwd():
        indexer_bwd_interface(index_q, weights, index_k, attn_score_base, index_score_base, topk_indices_base, offsets)

    def run_dsa_bwd():
        _run_dsa_backward(
            q,
            kv,
            index_q,
            index_k,
            weights,
            offsets,
            topk_indices_base,
            index_score_base,
            o_base,
            lse_base,
            do,
            args.dim,
            args.sm_scale,
        )

    with torch.no_grad():
        fwd_indexer_ms = _time_cuda_events(run_indexer_fwd, args.iters)
        fwd_total_ms = _time_cuda_events(run_dsa_fwd, args.iters)
        bwd_indexer_ms = _time_cuda_events(run_indexer_bwd, args.iters)
        bwd_total_ms = _time_cuda_events(run_dsa_bwd, args.iters)

    print("CUDA event breakdown (ms/iter):")
    print(f"  fwd_total={fwd_total_ms:.3f}  indexer_fwd={fwd_indexer_ms:.3f}  ratio={fwd_indexer_ms / fwd_total_ms:.2%}")
    print(f"  bwd_total={bwd_total_ms:.3f}  indexer_bwd={bwd_indexer_ms:.3f}  ratio={bwd_indexer_ms / bwd_total_ms:.2%}")

    with torch.no_grad():
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_fwd:
            for _ in range(args.iters):
                with record_function("indexer_fwd"):
                    topk_indices_fwd, index_score_fwd = indexer_topk_reducesum_interface(
                        index_q, weights, index_k, args.topk, offsets
                    )
                with record_function("mla_fwd"):
                    sparse_mla_fwd_interface(
                        q,
                        kv.unsqueeze(-2),
                        topk_indices_fwd.unsqueeze(-2),
                        offsets,
                        sm_scale=args.sm_scale,
                        d_v=args.dim,
                    )
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_bwd:
            for _ in range(args.iters):
                with record_function("topk_reducesum"):
                    attn_score = sparse_mla_topk_reducesum_interface(
                        q,
                        kv.unsqueeze(-2),
                        topk_indices_base.unsqueeze(-2),
                        lse_base,
                        offsets,
                        dim_v=args.dim,
                    ).squeeze(-2)
                with record_function("mla_bwd"):
                    sparse_mla_bwd(
                        q,
                        kv.unsqueeze(-2),
                        o_base,
                        do,
                        topk_indices_base.unsqueeze(-2),
                        lse_base,
                        offsets,
                        sm_scale=args.sm_scale,
                    )
                with record_function("indexer_bwd"):
                    indexer_bwd_interface(
                        index_q, weights, index_k, attn_score, index_score_base, topk_indices_base, offsets
                    )
        torch.cuda.synchronize()

    fwd_indexer_ms = _get_profiler_ms(prof_fwd, "indexer_fwd")
    fwd_mla_ms = _get_profiler_ms(prof_fwd, "mla_fwd")
    fwd_total_ms = fwd_indexer_ms + fwd_mla_ms

    bwd_topk_ms = _get_profiler_ms(prof_bwd, "topk_reducesum")
    bwd_mla_ms = _get_profiler_ms(prof_bwd, "mla_bwd")
    bwd_indexer_ms = _get_profiler_ms(prof_bwd, "indexer_bwd")
    bwd_total_ms = bwd_topk_ms + bwd_mla_ms + bwd_indexer_ms

    print("torch.profiler breakdown (total ms over all iters):")
    print(f"  fwd_total={fwd_total_ms:.3f}  indexer_fwd={fwd_indexer_ms:.3f}  ratio={fwd_indexer_ms / fwd_total_ms:.2%}")
    print(f"  bwd_total={bwd_total_ms:.3f}  indexer_bwd={bwd_indexer_ms:.3f}  ratio={bwd_indexer_ms / bwd_total_ms:.2%}")


if __name__ == "__main__":
    main()
