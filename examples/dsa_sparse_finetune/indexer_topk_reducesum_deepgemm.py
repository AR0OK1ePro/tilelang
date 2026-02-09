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

pass_configs = {
    tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


# MOD: deepgemm helper
def ceildiv(a, b):
    return (a + b - 1) // b


# MOD: deepgemm helper
def cast_q_token_scale(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    seq_len = x.shape[0]
    x_amax = x.abs().float().amax(dim=(1, 2)).clamp(1e-4)
    x_scaled = (x * (448.0 / x_amax.view(seq_len, 1, 1))).to(torch.float8_e4m3fn)
    return x_scaled, (x_amax / 448.0)


def cast_k_token_scale(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    x_amax = x.abs().float().amax(dim=1).clamp(1e-4)
    x_scaled = (x * (448.0 / x_amax.view(-1, 1))).to(torch.float8_e4m3fn)
    return x_scaled, (x_amax / 448.0)


def _sum_ceil_1_to_n(n: int, d: int) -> int:
    q, r = divmod(n, d)
    return d * q * (q + 1) // 2 + r * (q + 1)


def _sum_floor_0_to_n_minus_1(n: int, d: int) -> int:
    q, r = divmod(n, d)
    return d * q * (q - 1) // 2 + q * r


def estimate_indexer_flops(
    seq_len: int,
    heads: int,
    dim: int,
    topk: int,
    block_k: int = 128,
):
    sum_blocks = _sum_ceil_1_to_n(seq_len, block_k)
    block_elems = block_k * sum_blocks

    # 1) Main GEMM FLOPs actually executed by this kernel (includes block padding)
    gemm_executed_flops = 2 * dim * heads * block_elems
    # 2) Causal-valid GEMM FLOPs (ideal useful work, no padding)
    gemm_useful_flops = heads * dim * seq_len * (seq_len + 1)

    # Non-GEMM float ops in this kernel (rough estimate, excludes exp/max/compare)
    scale_logits_flops = 2 * block_elems
    post_relu_weight_flops = 2 * heads * block_elems
    reduce_sum_flops = (heads - 1) * block_elems
    softmax_shift_flops = seq_len * topk  # x - max
    softmax_reduce_sum_flops = seq_len * (topk - 1)
    softmax_div_flops = seq_len * topk
    approx_total_flops_no_exp = (
        gemm_executed_flops
        + scale_logits_flops
        + post_relu_weight_flops
        + reduce_sum_flops
        + softmax_shift_flops
        + softmax_reduce_sum_flops
        + softmax_div_flops
    )

    # Bitonic sort compare count (not FLOPs, but often dominates runtime for large topk)
    n = 2 * topk
    num_iters = int(round(math.log2(n)))
    compares_per_sort = (n // 2) * (num_iters * (num_iters + 1) // 2)
    sort_calls_per_token_total = seq_len + _sum_floor_0_to_n_minus_1(seq_len, topk)
    bitonic_compare_ops = compares_per_sort * sort_calls_per_token_total

    return {
        "sum_blocks": sum_blocks,
        "gemm_executed_flops": gemm_executed_flops,
        "gemm_useful_flops": gemm_useful_flops,
        "approx_total_flops_no_exp": approx_total_flops_no_exp,
        "bitonic_compare_ops": bitonic_compare_ops,
    }


def bench_kernel_ms(kernel, kernel_args, warmup: int = 10, rep: int = 20) -> float:
    for _ in range(warmup):
        kernel(*kernel_args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        kernel(*kernel_args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


@tl.jit(pass_configs=pass_configs)
def tl_indexer_topk_reducesum_impl(
    heads: int,
    dim: int,
    topk: int,
    seq_len: int = 163840,
    sm_scale: Optional[float] = None,
    block_K: int = 128,
    dtype: str = FP32,
    num_stages: int = 0,
    num_threads: int = 128,
):
    assert topk == tl.math.next_power_of_2(topk)
    assert topk % block_K == 0
    assert heads <= 64 and heads % 8 == 0
    assert num_stages == 0
    batch_plus_one = 2

    weights_shape = [seq_len, heads]
    index_q_fp8_shape = [seq_len, heads, dim]  # MOD: deepgemm FP8 Q
    index_k_fp8_shape = [seq_len, dim]  # MOD: deepgemm FP8 K
    scale_q_shape = [seq_len]  # MOD: per-token Q scale
    scale_k_shape = [seq_len]  # MOD: per-token K scale
    topk_indices_shape = [seq_len, topk]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]

    N = 2 * topk
    num_iters = int(round(math.log2(N)))
    if sm_scale is None:
        sm_scale = dim**-0.5

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
        IndexQFp8: T.Tensor(index_q_fp8_shape, FP8),  # MOD: deepgemm FP8 Q
        IndexKFp8: T.Tensor(index_k_fp8_shape, FP8),  # MOD: deepgemm FP8 K
        ScaleQ: T.Tensor(scale_q_shape, FP32),  # MOD: deepgemm scale Q
        ScaleK: T.Tensor(scale_k_shape, FP32),  # MOD: deepgemm scale K
        Weights: T.Tensor(weights_shape, dtype),
        TopkIndices: T.Tensor(topk_indices_shape, INT32),
        ReduceSum: T.Tensor(topk_indices_shape, FP32),
        Offsets: T.Tensor(offsets_shape, INT32),
        TokenIndices: T.Tensor(token_indices_shape, INT32),
    ):
        with T.Kernel(seq_len, threads=num_threads) as (bx):
            i_b, i_t = TokenIndices[bx, 0], TokenIndices[bx, 1]
            bos, eos = Offsets[i_b], Offsets[i_b + 1]
            num_blocks = T.ceildiv(i_t + 1, block_K)

            topk_index_shared = T.alloc_shared([N], dtype=INT32)
            topk_value_shared = T.alloc_shared([N], dtype=FP32)

            T.fill(topk_index_shared, -1)
            T.fill(topk_value_shared, float("-inf"))
            T.sync_threads()

            indexQFp8_shared = T.alloc_shared([heads, dim], dtype=FP8)
            T.copy(IndexQFp8[bos + i_t, :, :], indexQFp8_shared)
            T.sync_threads()

            weights_frag = T.alloc_shared([heads], dtype=dtype)
            T.copy(Weights[bos + i_t, :], weights_frag)
            T.sync_threads()

            for bk_i in T.Pipelined(num_blocks, num_stages=num_stages):
                k_st = bk_i * block_K
                k_ed = T.min((bk_i + 1) * block_K, eos - bos)

                indexKFp8_shared = T.alloc_shared([block_K, dim], dtype=FP8)
                for i, j in T.Parallel(block_K, dim):
                    indexKFp8_shared[i, j] = T.if_then_else(k_st + i < k_ed, IndexKFp8[bos + k_st + i, j], 0)
                T.sync_threads()

                # MOD: deepgemm FP8 path
                scale_logits_shared = T.alloc_shared([block_K], dtype=FP32)
                for i in T.Parallel(block_K):
                    scale_logits_shared[i] = T.if_then_else(k_st + i < k_ed, ScaleK[bos + k_st + i] * ScaleQ[bos + i_t] * sm_scale, 0)
                T.sync_threads()

                logits = T.alloc_fragment((block_K, heads), FP32)
                T.gemm(
                    indexKFp8_shared,
                    indexQFp8_shared,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=True,
                    )
                T.sync_threads()

                for i, j in T.Parallel(block_K, heads):
                    logits[i, j] = T.max(logits[i, j] * scale_logits_shared[i], 0) * weights_frag[j]
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
                    bitonic_sort(topk_index_shared, topk_value_shared)

            bitonic_sort(topk_index_shared, topk_value_shared)

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
    enable_profile: bool = False,
    profile_warmup: int = 10,
    profile_rep: int = 20,
):
    seq_len, heads, dim = q.shape
    kernel = tl_indexer_topk_reducesum_impl(heads=heads, dim=dim, topk=topk, seq_len=seq_len, dtype=dtype)
    token_indices = prepare_token_indices(offsets)
    topk_indices = torch.zeros((seq_len, topk), device=q.device, dtype=torch.int32)
    topk_score = torch.zeros((seq_len, topk), device=q.device, dtype=torch.float32)
    # MOD: FP8 quantization for deepgemm
    q_fp8, scale_q = cast_q_token_scale(q)
    k_fp8, scale_k = cast_k_token_scale(k)
    kernel(q_fp8, k_fp8, scale_q, scale_k, weights, topk_indices, topk_score, offsets, token_indices)

    if enable_profile:
        latency = bench_kernel_ms(
            kernel,
            (q_fp8, k_fp8, scale_q, scale_k, weights, topk_indices, topk_score, offsets, token_indices),
            warmup=profile_warmup,
            rep=profile_rep,
        )
        stat = estimate_indexer_flops(seq_len, heads, dim, topk)

        tflops_executed_gemm = stat["gemm_executed_flops"] / latency / 1e9
        tflops_useful_gemm = stat["gemm_useful_flops"] / latency / 1e9
        tflops_total_no_exp = stat["approx_total_flops_no_exp"] / latency / 1e9

        print(f"kernel latency (kernel-only, warmup={profile_warmup}, rep={profile_rep}): {latency:.3f} ms")
        print(f"executed GEMM FLOPs/call: {stat['gemm_executed_flops']}")
        print(f"useful GEMM FLOPs/call (causal valid): {stat['gemm_useful_flops']}")
        print(f"approx total FLOPs/call (no exp/max/sort): {stat['approx_total_flops_no_exp']}")
        print(f"approx bitonic compare ops/call: {stat['bitonic_compare_ops']}")
        print(f"throughput (executed GEMM): {tflops_executed_gemm:.3f} TFLOP/s")
        print(f"throughput (useful causal GEMM): {tflops_useful_gemm:.3f} TFLOP/s")
        print(f"throughput (approx total no-exp): {tflops_total_no_exp:.3f} TFLOP/s")

    return topk_indices, topk_score


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
    H=64,
    D=128,
    cases: Optional[list[Tuple[int, int]]] = None,
):
    del B  # unused, keep signature backward-compatible
    if cases is None:
        cases = [
            (2048, 128),
            (4096, 256),
            (8192, 256),
            (16384, 512),
            (32768, 512),
            (65536, 1024),
            (98304, 1024),
            (131072, 2048),
            (163840, 2048),
        ]

    for i, (S, topk) in enumerate(cases, start=1):
        if topk > S:
            print(f"[case {i}/{len(cases)}] skip S={S}, topk={topk}: topk must be <= S")
            continue
        if topk != tl.math.next_power_of_2(topk) or topk % 128 != 0:
            print(f"[case {i}/{len(cases)}] skip S={S}, topk={topk}: topk must be power-of-2 and divisible by 128")
            continue

        # Keep long-shape sweep practical.
        if S >= 131072:
            warmup, rep = 1, 2
        elif S >= 65536:
            warmup, rep = 1, 3
        elif S >= 16384:
            warmup, rep = 2, 5
        else:
            warmup, rep = 3, 8

        print(f"\n[case {i}/{len(cases)}] S={S}, topk={topk}, H={H}, D={D}, warmup={warmup}, rep={rep}")
        torch.manual_seed(42)
        q = torch.randn((S, H, D)).cuda().bfloat16()
        weights = torch.randn((S, H)).cuda().bfloat16()
        k = torch.randn((S, D)).cuda().bfloat16()
        offsets = torch.tensor([0, S], dtype=torch.int32).cuda()

        # ref_topk_indices, ref_topk_score = ref_index_score(q, weights, k, topk, offsets)
        topk_indices, topk_score = indexer_topk_reducesum_interface(
            q,
            weights,
            k,
            topk,
            offsets,
            enable_profile=True,
            profile_warmup=warmup,
            profile_rep=rep,
        )

        del q, weights, k, offsets, topk_indices, topk_score
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # for j in range(S):
        # ref_np = ref_topk_indices[j].cpu().to(torch.int32).numpy()
        # trt_np = topk_indices[j].cpu().to(torch.int32).numpy()

        # ref_np_val = ref_topk_score[j]
        # trt_np_val = topk_score[j]

        # mask = (ref_np_val > 0).cpu().numpy()

        # set_ref = set(ref_np[mask])
        # set_trt = set(trt_np[mask])
        # intersection = set_ref & set_trt

        # print("idx:", j, "selected/all:", len(intersection), "/", len(set_ref), "=", len(intersection) / len(set_ref))

        # print(f"err: {get_abs_err(ref_np_val, trt_np_val):.6f} ratio: {get_err_ratio(ref_np_val, trt_np_val):.6f}")


if __name__ == "__main__":
    test_kernel()
