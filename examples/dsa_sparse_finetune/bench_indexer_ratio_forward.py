import torch
from tilelang.profiler import do_bench

from indexer_topk_reducesum import indexer_topk_reducesum_interface
from sparse_mla_fwd import sparse_mla_fwd_interface


def build_inputs(
    *,
    seq_len: int,
    heads: int,
    dim: int,
    tail_dim: int,
    index_dim: int,
    topk: int,
):
    torch.manual_seed(0)
    q = torch.randn((seq_len, heads, dim + tail_dim), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((seq_len, dim + tail_dim), device="cuda", dtype=torch.bfloat16)
    index_q = torch.randn((seq_len, heads, index_dim), device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn((seq_len, index_dim), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((seq_len, heads), device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    return q, kv, index_q, index_k, weights, offsets, topk, dim


def forward_once(inputs):
    q, kv, index_q, index_k, weights, offsets, topk, dim = inputs
    torch.cuda.nvtx.range_push("dsa/forward_total")
    torch.cuda.nvtx.range_push("dsa/indexer_topk")
    topk_indices, _ = indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("dsa/sparse_mla_fwd")
    result = sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, d_v=dim)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    return result


def measure_with_do_bench(inputs, *, warmup=50, rep=200):
    t_total = do_bench(lambda: forward_once(inputs), warmup=warmup, rep=rep)
    return t_total


def time_cuda(fn, *, warmup=10, rep=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def measure_with_cuda_events(inputs, *, warmup=10, rep=200):
    for _ in range(warmup):
        forward_once(inputs)
    torch.cuda.synchronize()

    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    indexer_start = torch.cuda.Event(enable_timing=True)
    indexer_end = torch.cuda.Event(enable_timing=True)

    total_start.record()
    indexer_start.record()
    q, kv, index_q, index_k, weights, offsets, topk, dim = inputs
    topk_indices, _ = indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)
    indexer_end.record()
    sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, d_v=dim)
    total_end.record()
    torch.cuda.synchronize()

    t_indexer = indexer_start.elapsed_time(indexer_end)
    t_total = total_start.elapsed_time(total_end)
    ratio = t_indexer / t_total if t_total > 0 else 0.0
    return t_indexer, t_total, ratio


def print_result(label, t_indexer, t_total, ratio):
    print(f"[{label}] Indexer+TopK: {t_indexer:.3f} ms")
    print(f"[{label}] Total forward: {t_total:.3f} ms")
    print(f"[{label}] Indexer ratio: {ratio * 100:.2f}%")


def main():
    inputs = build_inputs(
        seq_len=163840,
        heads=64,
        dim=512,
        tail_dim=64,
        index_dim=128,
        topk=2048,
    )

    t_total = measure_with_do_bench(inputs)
    print(f"[do_bench] Total forward: {t_total:.3f} ms")

    t_indexer, t_total, ratio = measure_with_cuda_events(inputs)
    print_result("cuda_event", t_indexer, t_total, ratio)

    print("\n[nsys] Example command:")
    print("  nsys profile --trace=cuda,nvtx --force-overwrite true -o /data/dsa_indexer_ratio python3 ./examples/dsa_sparse_finetune/bench_indexer_ratio_forward.py")


if __name__ == "__main__":
    main()
