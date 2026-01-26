import math
import torch
import torch.nn.functional as F
from einops import einsum

import tilelang as tl
import tilelang.language as T
from typing import Optional, Tuple  # MOD: add Tuple for FP8 helpers
from index import prepare_token_indices

from utils import get_abs_err, get_err_ratio

BF16 = T.bfloat16
FP32 = T.float32
INT32 = T.int32
FP8 = T.float8_e4m3fn  # MOD: deepgemm FP8
INT64 = T.int64  # MOD: timing counters

pass_configs = {
    tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


# MOD: deepgemm helper
def ceildiv(a, b):
    return (a + b - 1) // b


# MOD: deepgemm helper
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


# MOD: deepgemm helper
def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(ceildiv(m, 128) * 128, ceildiv(n, 128) * 128, dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


# MOD: deepgemm helper
def per_block_cast_to_fp8_3d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    seq_len = x.shape[0]
    fp8_chunks = []
    scale_chunks = []
    for i in range(seq_len):
        fp8_i, scale_i = per_block_cast_to_fp8(x[i])
        fp8_chunks.append(fp8_i)
        scale_chunks.append(scale_i.squeeze(0))
    return torch.stack(fp8_chunks, dim=0), torch.stack(scale_chunks, dim=0)


@tl.jit(pass_configs=pass_configs)
def tl_indexer_topk_reducesum_impl(
    heads: int,
    dim: int,
    topk: int,
    sm_scale: Optional[float] = None,
    block_K: int = 32,
    dtype: str = FP32,
    num_stages: int = 0,
    num_threads: int = 128,
    use_deepgemm: bool = False,  # MOD: choose deepgemm vs T.gemm
):
    assert topk == tl.math.next_power_of_2(topk)
    assert topk % block_K == 0
    assert heads <= 64 and heads % 8 == 0
    assert num_stages == 0
    if use_deepgemm:
        assert dim % 128 == 0  # MOD: deepgemm requires K aligned
        assert topk % 128 == 0  # MOD: per request, enforce topk alignment
    batch_plus_one = T.symbolic("batch_plus_one")
    seq_len = T.symbolic("seq_len")

    index_q_shape = [seq_len, heads, dim]
    weights_shape = [seq_len, heads]
    index_k_shape = [seq_len, dim]
    index_q_fp8_shape = [seq_len, heads, dim]  # MOD: deepgemm FP8 Q
    index_k_fp8_shape = [seq_len, dim]  # MOD: deepgemm FP8 K
    dim_blocks = ceildiv(dim, 128)  # MOD: scale blocks
    block_D = 128  # MOD: deepgemm K tile
    scale_q_shape = [seq_len, dim_blocks]  # MOD: deepgemm Q scale
    scale_k_shape = [seq_len, dim_blocks]  # MOD: deepgemm K scale
    topk_indices_shape = [seq_len, topk]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    timing_shape = [1]  # MOD: timing counters

    N = 2 * topk
    num_iters = int(round(math.log2(N)))
    if sm_scale is None:
        sm_scale = dim**-0.5

    # MOD: timing helper
    @T.macro
    def read_clock():
        return T.call_llvm_intrin("int64", "llvm.nvvm.read.ptx.sreg.clock64")

    @T.macro
    def bitonic_sort(
        topk_index_shared: T.SharedBuffer([N], dtype=INT32),
        topk_value_shared: T.SharedBuffer([N], dtype=FP32),
    ):
        T.sync_threads()
        for i1 in T.serial(num_iters):
            for i2 in T.serial(i1 + 1):
                for i in T.Parallel(N):
                    ascending = (i & (1 << (i1 + 1))) != 0
                    j = i ^ (1 << (i1 - i2))
                    if i < j and (
                        (ascending and topk_value_shared[i] > topk_value_shared[j])
                        or (not ascending and topk_value_shared[i] < topk_value_shared[j])
                    ):
                        val = topk_value_shared[i]
                        topk_value_shared[i] = topk_value_shared[j]
                        topk_value_shared[j] = val
                        idx = topk_index_shared[i]
                        topk_index_shared[i] = topk_index_shared[j]
                        topk_index_shared[j] = idx
                T.sync_threads()

    @T.prim_func
    def tl_indexer_topk_reducesum_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),
        Weights: T.Tensor(weights_shape, dtype),
        IndexK: T.Tensor(index_k_shape, dtype),
        IndexQFp8: T.Tensor(index_q_fp8_shape, FP8),  # MOD: deepgemm FP8 Q
        IndexKFp8: T.Tensor(index_k_fp8_shape, FP8),  # MOD: deepgemm FP8 K
        ScaleQ: T.Tensor(scale_q_shape, FP32),  # MOD: deepgemm scale Q
        ScaleK: T.Tensor(scale_k_shape, FP32),  # MOD: deepgemm scale K
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        ReduceSum: T.Tensor(topk_indices_shape, FP32),
        Offsets: T.Tensor(offsets_shape, INT32),
        TokenIndices: T.Tensor(token_indices_shape, INT32),
        GemmCycles: T.Tensor(timing_shape, INT64),  # MOD: gemm timing
        TopkCycles: T.Tensor(timing_shape, INT64),  # MOD: topk timing
    ):
        with T.Kernel(seq_len, threads=num_threads) as (bx):
            thread_id = T.get_thread_binding()  # MOD: timing guard
            i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
            bos, eos = Offsets[i_b], Offsets[i_b + 1]
            num_blocks = T.ceildiv(i_t + 1, block_K)

            topk_index_shared = T.alloc_shared([N], dtype=INT32)
            topk_value_shared = T.alloc_shared([N], dtype=FP32)

            T.fill(topk_index_shared, -1)
            T.fill(topk_value_shared, float("-inf"))
            T.sync_threads()

            index_q_shared = T.alloc_shared([heads, dim], dtype=dtype)
            T.copy(IndexQ[bos + i_t, :, :], index_q_shared)
            T.sync_threads()

            weights_frag = T.alloc_shared([heads], dtype=dtype)
            T.copy(Weights[bos + i_t, :], weights_frag)
            T.sync_threads()

            for i, j in T.Parallel(heads, dim):
                index_q_shared[i, j] = index_q_shared[i, j] * sm_scale
            T.sync_threads()

            for bk_i in T.Pipelined(num_blocks, num_stages=num_stages):
                k_st = bk_i * block_K
                k_ed = T.min((bk_i + 1) * block_K, eos - bos)

                index_k_shared = T.alloc_shared([block_K, dim], dtype=dtype)
                for i, j in T.Parallel(block_K, dim):
                    index_k_shared[i, j] = T.if_then_else(k_st + i < k_ed, IndexK[bos + k_st + i, j], 0)
                T.sync_threads()

                logits = T.alloc_fragment((block_K, heads), FP32)
                if use_deepgemm:
                    # MOD: deepgemm FP8 path
                    T.clear(logits)
                    k_fp8_shared = T.alloc_shared([block_K, block_D], dtype=FP8)
                    q_fp8_shared = T.alloc_shared([heads, block_D], dtype=FP8)
                    scale_k_shared = T.alloc_shared([block_K], dtype=FP32)
                    gemm_tmp = T.alloc_fragment((block_K, heads), FP32)
                    for kd in T.serial(dim_blocks):
                        for i, j in T.Parallel(block_K, block_D):
                            k_fp8_shared[i, j] = T.if_then_else(
                                k_st + i < k_ed,
                                IndexKFp8[bos + k_st + i, kd * block_D + j],
                                0,
                            )
                        for i, j in T.Parallel(heads, block_D):
                            q_fp8_shared[i, j] = IndexQFp8[bos + i_t, i, kd * block_D + j]
                        for i in T.Parallel(block_K):
                            scale_k_shared[i] = T.if_then_else(k_st + i < k_ed, ScaleK[bos + k_st + i, kd], 0)
                        T.sync_threads()

                        gemm_start = T.alloc_var(INT64)
                        gemm_end = T.alloc_var(INT64)
                        T.sync_threads()
                        if thread_id == 0:
                            gemm_start = read_clock()
                        T.sync_threads()
                        T.gemm(
                            k_fp8_shared,
                            q_fp8_shared,
                            gemm_tmp,
                            transpose_A=False,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.sync_threads()
                        if thread_id == 0:
                            gemm_end = read_clock()
                            T.atomic_add(GemmCycles[0], gemm_end - gemm_start)
                        T.sync_threads()

                        scale_q = ScaleQ[bos + i_t, kd] * sm_scale
                        for i, j in T.Parallel(block_K, heads):
                            logits[i, j] += gemm_tmp[i, j] * (scale_k_shared[i] * scale_q)
                        T.sync_threads()
                else:
                    # MOD: timing around regular gemm
                    gemm_start = T.alloc_var(INT64)
                    gemm_end = T.alloc_var(INT64)
                    T.sync_threads()
                    if thread_id == 0:
                        gemm_start = read_clock()
                    T.sync_threads()
                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        logits,
                        transpose_A=False,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.sync_threads()
                    if thread_id == 0:
                        gemm_end = read_clock()
                        T.atomic_add(GemmCycles[0], gemm_end - gemm_start)
                    T.sync_threads()

                for i, j in T.Parallel(block_K, heads):
                    logits[i, j] = T.max(logits[i, j], 0) * weights_frag[j]
                T.sync_threads()

                logits_sum = T.alloc_fragment(block_K, FP32)
                T.reduce_sum(logits, logits_sum, dim=1)
                T.sync_threads()

                offset = T.alloc_var(INT32)
                if k_st >= topk:
                    offset = topk + (k_st % topk)
                else:
                    offset = k_st
                T.sync_threads()
                for i in T.Parallel(block_K):
                    if k_st + i > i_t:
                        logits_sum[i] = float("-inf")
                    j = offset + i
                    topk_index_shared[j] = k_st + i
                    topk_value_shared[j] = logits_sum[i]
                T.sync_threads()

                if k_ed > topk and k_ed % topk == 0:
                    # MOD: timing around topk (bitonic_sort)
                    topk_start = T.alloc_var(INT64)
                    topk_end = T.alloc_var(INT64)
                    T.sync_threads()
                    if thread_id == 0:
                        topk_start = read_clock()
                    T.sync_threads()
                    bitonic_sort(topk_index_shared, topk_value_shared)
                    T.sync_threads()
                    if thread_id == 0:
                        topk_end = read_clock()
                        T.atomic_add(TopkCycles[0], topk_end - topk_start)
                    T.sync_threads()

            # MOD: timing around topk (bitonic_sort)
            topk_start = T.alloc_var(INT64)
            topk_end = T.alloc_var(INT64)
            T.sync_threads()
            if thread_id == 0:
                topk_start = read_clock()
            T.sync_threads()
            bitonic_sort(topk_index_shared, topk_value_shared)
            T.sync_threads()
            if thread_id == 0:
                topk_end = read_clock()
                T.atomic_add(TopkCycles[0], topk_end - topk_start)
            T.sync_threads()

            logits_max_frag = T.alloc_fragment([1], dtype=FP32)
            logits_frag = T.alloc_fragment([topk], dtype=FP32)
            reducesum_shared = T.alloc_shared([topk], dtype=FP32)

            T.copy(topk_value_shared[:topk], logits_frag)
            T.sync_threads()

            T.reduce_max(logits_frag, logits_max_frag, dim=-1)
            T.sync_threads()

            for i in T.Parallel(topk):
                logits_frag[i] = T.exp(logits_frag[i] - logits_max_frag[0])
            T.sync_threads()

            lse_frag = T.alloc_fragment([1], dtype=FP32)
            T.reduce_sum(logits_frag, lse_frag)
            T.sync_threads()

            for i in T.Parallel(topk):
                reducesum_shared[i] = logits_frag[i] / lse_frag[0]
            T.sync_threads()

            # for i in T.Parallel(topk):
            #     reducesum_shared[i] = logits_frag[i]
            # T.sync_threads()

            for i in T.Parallel(topk):
                if topk_index_shared[i] > i_t:
                    topk_index_shared[i] = -1
            T.sync_threads()

            T.copy(topk_index_shared[:topk], TopkIndices[bos + i_t, :])
            T.copy(reducesum_shared[:topk], ReduceSum[bos + i_t, :])

    return tl_indexer_topk_reducesum_kernel


def indexer_topk_reducesum_interface(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    topk: int,
    offsets: torch.Tensor,
    dtype: str = BF16,
    use_deepgemm: bool = False,  # MOD: choose gemm path
):
    seq_len, heads, dim = q.shape
    kernel = tl_indexer_topk_reducesum_impl(heads=heads, dim=dim, topk=topk, dtype=dtype, use_deepgemm=use_deepgemm)
    token_indices = prepare_token_indices(offsets)
    topk_indices = torch.zeros((seq_len, topk), device=q.device, dtype=torch.int32)
    topk_score = torch.zeros((seq_len, topk), device=q.device, dtype=torch.float32)
    gemm_cycles = torch.zeros((1,), device=q.device, dtype=torch.int64)  # MOD: gemm timing
    topk_cycles = torch.zeros((1,), device=q.device, dtype=torch.int64)  # MOD: topk timing
    dim_blocks = ceildiv(dim, 128)  # MOD: scale blocks
    if use_deepgemm:
        # MOD: FP8 quantization for deepgemm
        q_fp8, scale_q = per_block_cast_to_fp8_3d(q)
        k_fp8, scale_k = per_token_cast_to_fp8(k)
    else:
        # MOD: placeholders when deepgemm is disabled
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        k_fp8 = torch.empty_like(k, dtype=torch.float8_e4m3fn)
        scale_q = torch.empty((seq_len, dim_blocks), device=q.device, dtype=torch.float32)
        scale_k = torch.empty((seq_len, dim_blocks), device=q.device, dtype=torch.float32)
    kernel(q, weights, k, q_fp8, k_fp8, scale_q, scale_k, topk_indices, topk_score, offsets, token_indices, gemm_cycles, topk_cycles)
    return topk_indices, topk_score, gemm_cycles, topk_cycles


def ref_index_score(Q: torch.Tensor, Weights: torch.Tensor, K: torch.Tensor, topk: int, offsets: torch.Tensor) -> torch.Tensor:
    all_topk_indices = []
    all_topk_score = []
    for i in range(offsets.shape[0] - 1):
        assert (offsets[i + 1] - offsets[i]).item() >= topk
        q = Q[offsets[i] : offsets[i + 1]]
        weights = Weights[offsets[i] : offsets[i + 1]]
        k = K[offsets[i] : offsets[i + 1]]
        softmax_scale = q.shape[-1] ** -0.5
        s = q.shape[0]
        mask = (torch.arange(s)[:, None] >= torch.arange(s)[None, :]).to(q.device)
        logits = einsum(q, k, "s1 h k, s2 k -> s1 h s2")
        logits = F.relu(logits)
        logits = (logits * weights.unsqueeze(-1)).sum(dim=-2, dtype=torch.float32) * softmax_scale
        logits = torch.where(mask, logits, float("-inf"))
        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        topk_score = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
        all_topk_indices.append(topk_indices)
        all_topk_score.append(topk_score)
    topk_indices = torch.cat(all_topk_indices, dim=0)
    topk_score = torch.cat(all_topk_score, dim=0)
    return topk_indices, topk_score


def test_kernel(
    B=1,
    S=2048,
    H=64,
    D=128,
    topk=64,
):
    torch.manual_seed(42)

    q = torch.randn((S, H, D)).cuda().bfloat16()
    weights = torch.randn((S, H)).cuda().bfloat16()
    k = torch.randn((S, D)).cuda().bfloat16()
    offsets = torch.tensor([0, S], dtype=torch.int32).cuda()

    ref_topk_indices, ref_topk_score = ref_index_score(q, weights, k, topk, offsets)

    # MOD: capture timing outputs
    topk_indices, topk_score, _, _ = indexer_topk_reducesum_interface(q, weights, k, topk, offsets)

    for j in range(S):
        ref_np = ref_topk_indices[j].cpu().to(torch.int32).numpy()
        trt_np = topk_indices[j].cpu().to(torch.int32).numpy()

        ref_np_val = ref_topk_score[j]
        trt_np_val = topk_score[j]

        mask = (ref_np_val > 0).cpu().numpy()

        set_ref = set(ref_np[mask])
        set_trt = set(trt_np[mask])
        intersection = set_ref & set_trt

        print("idx:", j, "selected/all:", len(intersection), "/", len(set_ref), "=", len(intersection) / len(set_ref))

        print(f"err: {get_abs_err(ref_np_val, trt_np_val):.6f} ratio: {get_err_ratio(ref_np_val, trt_np_val):.6f}")


if __name__ == "__main__":
    test_kernel()
