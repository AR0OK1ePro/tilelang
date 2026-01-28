import torch

from indexer_topk_reducesum import indexer_topk_reducesum_interface
from sparse_mla_fwd import sparse_mla_fwd_interface


def main():
    # ---- 配置 ----
    S = 163840
    H = 64
    D = 512
    tail_D = 64
    index_D = 128
    topk = 2048

    # ---- 输入 ----
    torch.manual_seed(0)
    q = torch.randn((S, H, D + tail_D), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((S, D + tail_D), device="cuda", dtype=torch.bfloat16)
    index_q = torch.randn((S, H, index_D), device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn((S, index_D), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((S, H), device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, S], dtype=torch.int32, device="cuda")

    # ---- 预热 ----
    for _ in range(10):
        topk_indices, _ = indexer_topk_reducesum_interface(
            index_q, weights, index_k, topk, offsets
        )
        sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, d_v=D)
    torch.cuda.synchronize()

    # ---- CUDA Events ----
    evt_total_start = torch.cuda.Event(enable_timing=True)
    evt_total_end = torch.cuda.Event(enable_timing=True)
    evt_idx_start = torch.cuda.Event(enable_timing=True)
    evt_idx_end = torch.cuda.Event(enable_timing=True)

    # ---- 单次 forward 计时 ----
    evt_total_start.record()

    evt_idx_start.record()
    topk_indices, _ = indexer_topk_reducesum_interface(
        index_q, weights, index_k, topk, offsets
    )
    evt_idx_end.record()

    sparse_mla_fwd_interface(q, kv.unsqueeze(-2), topk_indices.unsqueeze(-2), offsets, d_v=D)

    evt_total_end.record()
    torch.cuda.synchronize()

    # ---- 统计 ----
    t_indexer = evt_idx_start.elapsed_time(evt_idx_end)   # ms
    t_total = evt_total_start.elapsed_time(evt_total_end) # ms
    ratio = t_indexer / t_total if t_total > 0 else 0.0

    print(f"Indexer+TopK: {t_indexer:.3f} ms")
    print(f"Total forward: {t_total:.3f} ms")
    print(f"Indexer占比: {ratio * 100:.2f}%")

if __name__ == "__main__":
    main()