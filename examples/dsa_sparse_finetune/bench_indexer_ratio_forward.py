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
    topk_indices, _ = indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)
    return sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, d_v=dim)


def measure_with_do_bench(inputs, *, warmup=50, rep=200):
    def fn_indexer():
        q, kv, index_q, index_k, weights, offsets, topk, _ = inputs
        return indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)

    def fn_total():
        return forward_once(inputs)

    t_indexer = do_bench(fn_indexer, warmup=warmup, rep=rep)
    t_total = do_bench(fn_total, warmup=warmup, rep=rep)
    ratio = t_indexer / t_total if t_total > 0 else 0.0
    return t_indexer, t_total, ratio


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


def measure_with_cuda_events(inputs):
    q, kv, index_q, index_k, weights, offsets, topk, dim = inputs

    def fn_indexer():
        return indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)

    def fn_total():
        return forward_once(inputs)

    t_indexer = time_cuda(fn_indexer)
    t_total = time_cuda(fn_total)
    ratio = t_indexer / t_total if t_total > 0 else 0.0
    return t_indexer, t_total, ratio


def print_result(label, t_indexer, t_total, ratio):
    print(f"[{label}] Indexer+TopK: {t_indexer:.3f} ms")
    print(f"[{label}] Total forward: {t_total:.3f} ms")
    print(f"[{label}] Indexer ratio: {ratio * 100:.2f}%")


def main():
    inputs = build_inputs(
        seq_len=2048,
        heads=16,
        dim=512,
        tail_dim=64,
        index_dim=128,
        topk=64,
    )

    t_indexer, t_total, ratio = measure_with_do_bench(inputs)
    print_result("do_bench", t_indexer, t_total, ratio)

    t_indexer, t_total, ratio = measure_with_cuda_events(inputs)
    print_result("cuda_event", t_indexer, t_total, ratio)


if __name__ == "__main__":
    main()
