import torch
from tilelang.profiler import do_bench

from indexer_topk_reducesum import indexer_topk_reducesum_interface
from sparse_mla_fwd import sparse_mla_fwd_interface

def time_cuda(fn, warmup=10, rep=100):
    # warmup
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
    return start.elapsed_time(end) / rep  # ms

def main():
    # ---- 配置（可按需调整）----
    S = 2048
    H = 16
    D = 512
    tail_D = 64
    index_D = 128
    topk = 64

    # ---- 输入张量 ----
    torch.manual_seed(0)
    q = torch.randn((S, H, D + tail_D), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((S, D + tail_D), device="cuda", dtype=torch.bfloat16)
    index_q = torch.randn((S, H, index_D), device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn((S, index_D), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((S, H), device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, S], dtype=torch.int32, device="cuda")

    # ---- bench: indexer + topk ----
    def fn_indexer():
        return indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)

    # ---- bench: sparse mla fwd ----
    def fn_mla():
        topk_indices, _ = indexer_topk_reducesum_interface(index_q, weights, index_k, topk, offsets)
        return sparse_mla_fwd_interface(
            q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, d_v=D
        )

    print("Running benchmark by TileLang Profiler...")
    # 计时
    t_indexer = do_bench(fn_indexer, warmup=50, rep=200)
    t_mla = do_bench(fn_mla, warmup=50, rep=200)

    # 占比
    total = t_indexer + t_mla
    ratio = t_indexer / total
    print(f"Indexer+TopK time: {t_indexer:.3f} ms")
    print(f"Sparse MLA FWD time: {t_mla:.3f} ms")
    print(f"Indexer占比: {ratio * 100:.2f}%")

    print("\nRunning benchmark by manual CUDA timing...")
    # 计时
    t_indexer = time_cuda(fn_indexer, warmup=10, rep=200)
    t_mla = time_cuda(fn_mla, warmup=10, rep=200)

    total = t_indexer + t_mla
    ratio = t_indexer / total
    print(f"Indexer+TopK time: {t_indexer:.3f} ms")
    print(f"Sparse MLA FWD time: {t_mla:.3f} ms")
    print(f"Indexer占比: {ratio * 100:.2f}%")

if __name__ == "__main__":
    main()
