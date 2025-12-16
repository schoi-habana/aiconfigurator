# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# import tensorrt_llm
import torch
import torch.nn as nn
import torch.nn.functional as F
# from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
# from tensorrt_llm._torch.attention_backend.interface import (
#     AttentionRuntimeFeatures,
#     PositionalEmbeddingParams,
#     RopeParams,
# )
# from tensorrt_llm._torch.attention_backend.utils import create_attention
# from tensorrt_llm._torch.metadata import KVCacheParams
# from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
# from tensorrt_llm.functional import PositionEmbeddingType
# from tensorrt_llm.llmapi import KvCacheConfig
# from tensorrt_llm.mapping import Mapping
# from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from helper import log_perf #get_sm_version, 
from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV2 as FusedRoPE
from habana_frameworks.torch.hpex.kernels import FusedSDPA

class KVCache(torch.nn.Module):
    def __init__(self):
        super(KVCache, self).__init__()
        self.cache = None
        self.inp_seq_len = -1

    def allocate(self, inp_seq_len, dtype, device, shape):
        if self.cache is None or self.cache.shape != shape:
            self.inp_seq_len = inp_seq_len
            self.cache = torch.zeros(shape, dtype=dtype, device=device)
        else:
            assert self.inp_seq_len == inp_seq_len, (
                f"inp_seq_len must be the same. self.inp_seq_len:{self.inp_seq_len} inp_seq_len:{inp_seq_len}"
            )
            self.cache.fill_(0)

    @staticmethod
    def update(prev, cur, dim, idx, inp_seq_len):
        if inp_seq_len != -1:
            # reuse cache logic
            orig_cur = cur
            if prev.shape == cur.shape:
                prev.copy_(cur)
                return orig_cur
            if cur.shape[2] > 1 and cur.shape[2] <= prev.shape[2]:
                # Initialize
                prev[:, :, :inp_seq_len, :].copy_(cur)
                return orig_cur
        if idx is not None:
            # 2+ tokenizer logic if model is static shape optimized
            prev.index_copy_(dim, idx - 1, cur)
            return prev
        else:
            return torch.cat((prev, cur), dim=dim)

    def get_shape(self):
        if self.cache is None:
            return None
        return self.cache.shape

    def forward(self, cur, dim, idx):
        return self.update(self.cache, cur, dim, idx, self.inp_seq_len)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_dim, bs, inp_seq_len, dtype):
        super().__init__():
        self.k_cache = KVCache()
        self.v_cache = KVCache()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim

        self.rotary_emb = RotaryEmbedding(d=128).to(device)
        self.fsdpa = FusedSDPA
        
        cache_shape = (bs, self.num_kv_heads, inp_seq_len, self.head_dim)
        self.k_cache.allocate(inp_seq_len, dtype, device, cache_shape)
        self.v_cache.allocate(inp_seq_len, dtype, device, cache_shape)

    def forward(self, q_states, k_states, v_states, position_ids, is_context_phase):
        # hidden_states are Q,K,V after projection (#bs, #heads, sl, head_dim)

        cos, sin = self.rotary_emb(v_states, seq_len=seq_len)
        # q_states, k_states = apply_rope(q_states, k_states, cos, sin, position_ids)
        
        q_states = FusedRoPE.apply(
            q_states,
            cos.unsqueeze(0).unsqueeze(0).to(torch.float16),
            sin.unsqueeze(0).unsqueeze(0).to(torch.float16),
            position_ids,
        )
        k_states = FusedRoPE.apply(
            k_states,
            cos.unsqueeze(0).unsqueeze(0).to(torch.float16),
            sin.unsqueeze(0).unsqueeze(0).to(torch.float16),
            position_ids,
        )

        if not is_context_phase:
            self.k_cache(k_states, 2, 0)
            self.v_cache(v_states, 2, 0)
        
        if decode:
            self.fsdpa.apply(
                q_states,
                k_states,
                v_states,
                None,#attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
                softmax_mode="None",
                recompute_mode=False,
                valid_sequence_lengths=None,
                padding_side="left",
            )
        else:
            self.fsdpa.apply(
                q_states,
                k_states,
                v_states,
                None,
                dropout_p=0.0,
                is_causal=False,
                scale=None,
                softmax_mode="None",
                recompute_mode=False,
                valid_sequence_lengths=None,
                padding_side="left",
            )


def run_attention_torch(
    batch_size, #b
    input_len, #s
    num_heads, #n
    num_key_value_heads,  # keep same as num_heads for MHA
    head_dim,
    attention_window_size,
    # use_fp8_kv_cache,
    # use_fp8_context_fmha,
    is_context_phase,
    perf_filename,
    device="hpu",
):
    device = torch.device(device)
    # torch.set_default_device(device)
    # torch.cuda.set_device(device)

    # if XQA JIT is enabled, the context phase will also trigger XQA prepare which causes the error
    # with specifc q/kv head and seq setting.
    # if is_context_phase:
    #     os.environ["TRTLLM_ENABLE_XQA_JIT"] = "0"
    # else:
    #     os.environ["TRTLLM_ENABLE_XQA_JIT"] = "1"

    backend_name = "HPU"#TRTLLM"
    layer_idx = 0
    world_size = 1
    tp_size = 1
    tokens_per_block = 64
    warming_up = 10
    test_ite = 6
    output_len = 1
    # if use_fp8_context_fmha:
    #     assert use_fp8_kv_cache
    #     quant_algo = QuantAlgo.FP8
    #     out_scale = torch.tensor(
    #         [1.0],
    #         dtype=torch.float32,
    #         device=device,
    #     )  # fp8 fmha
    # else:
    #     quant_algo = None
    #     out_scale = None

    # if use_fp8_kv_cache:
    #     kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    # else:
    #     kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16

    # total_num_tokens = (input_len + output_len) * batch_size

    attn = Attention(num_heads, num_kv_heads, head_dim, batch_size, input_len + output_len, torch.bfloat16)

    # input_seq_lens = [input_len for _ in range(batch_size)]
    # total_seq_lens = [input_len + output_len for _ in range(batch_size)]
    # request_ids = list(range(batch_size))
    # kv_cache_manager.add_dummy_requests(request_ids, total_seq_lens)


    # sinks = torch.randn(num_heads, dtype=torch.float32) if head_dim == 64 else None
    if is_context_phase:
        q = torch.randn([batch_size, num_heads, input_len, head_dim]).bfloat16().to(device)
        k = torch.randn([batch_size, num_key_value_heads, input_len, head_dim]).bfloat16().to(device)
        v = torch.randn([batch_size, num_key_value_heads, input_len, head_dim]).bfloat16().to(device)
        position_ids = torch.arange(input_len)

        attn.forward(q, k, v, position_ids, is_context_phase)
    else:
        q = torch.randn([batch_size, num_heads, 1, head_dim]).bfloat16().to(device)
        k = torch.randn([batch_size, num_key_value_heads, 1, head_dim]).bfloat16().to(device)
        v = torch.randn([batch_size, num_key_value_heads, 1, head_dim]).bfloat16().to(device)
        position_ids = torch.arange(input_len)

        attn.forward(q, k, v, position_ids, is_context_phase)

    # warmup
    for i in range(warming_up):
        attn.forward(q, k, v, position_ids, is_context_phase)

    start_time = time.time()
    for i in range(test_ite):
        attn.forward(q, k, v, position_ids, is_context_phase)
    torch.hpu.synchronize()
    latency = (time.time() - start_time) / test_ite

    # write result
    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"
    kv_cache_dtype_str = "bfloat16"
    dtype = "bfloat16"
    # if use_fp8_kv_cache:
    #     kv_cache_dtype_str = "fp8"
    # if use_fp8_context_fmha:
    #     dtype_str = "fp8"
    # else:
    #     dtype_str = "float16"

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "window_size": attention_window_size,
                "beam_width": 1,
                "attn_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "step": step,
                "latency": latency,
            }
        ],
        framework="HPU",
        version=get_habana_fw_version(),
        device_name=get_device_name(),
        op_name=op_name,
        kernel_source="torch_compile",
        perf_filename=perf_filename,
    )
    kv_cache_manager.shutdown()


def get_context_attention_test_cases():
    # has_fp8 = get_sm_version() > 86
    test_cases = []
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    s_list = [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        10240,
        12288,
        16384,
        262144,
    ]
    n_list = [4, 8, 12, 16, 24, 32, 40, 48, 64, 96]
    n_kv_list = [0, 1, 2, 4, 8]
    head_dim = [64, 128]

    for h in head_dim:
        for n in sorted(n_list, reverse=True):
            for s in sorted(s_list, reverse=True):
                for b in sorted(b_list, reverse=True):
                    for n_kv in n_kv_list:
                        if n_kv != 0 and (n_kv >= n or n % n_kv != 0):
                            continue
                        num_kv_heads = n_kv if n_kv != 0 else n

                        if num_kv_heads == n:
                            if b * s > 65536 or b > 128:
                                continue
                        else:
                            if b * s > 131072:
                                continue
                        if b * s * num_kv_heads * 128 * 2 >= 2147483647:
                            continue
                        # if get_sm_version() >= 100:
                        #     # though it's a precheck of gen kernels during the attention op init,
                        #     # this cannot be skipped for now
                        #     # TLLM_CHECK_WITH_INFO((params.m_num_heads_q_per_kv < max_num_heads_q_per_kv_in_cta || params.m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta == 0), # noqa: E501
                        #     m_num_heads_q_per_kv = 1 if n_kv == 0 else n // n_kv
                        #     max_num_heads_q_per_kv_in_cta = 32
                        #     if (
                        #         m_num_heads_q_per_kv >= max_num_heads_q_per_kv_in_cta
                        #         and m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta != 0
                        #     ):
                        #         continue

                        # print(
                        #     f"collecting heads: {n} kv_heads: {num_kv_heads} seq: {s} "
                        #     f"batchsize: {b}"
                        # )
                        # use fp8 kv cache, fp8 context fmha, is_context_phase. in torch flow,
                        # int8 kvcache is not supported yet.
                        #
                        # fp16 kv cache, fp16 context fmha, is_context_phase
                        if h == 64:
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    num_kv_heads,
                                    h,
                                    128,
                                    False,
                                    False,
                                    True,
                                    "context_attention_perf.txt",
                                ]
                            )
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    num_kv_heads,
                                    h,
                                    0,
                                    False,
                                    False,
                                    True,
                                    "context_attention_perf.txt",
                                ]
                            )
                            if has_fp8:
                                # test_cases.append(
                                #     [
                                #         b,
                                #         s,
                                #         n,
                                #         num_kv_heads,
                                #         h,
                                #         128,
                                #         True,
                                #         False,
                                #         True,
                                #         "context_attention_perf.txt",
                                #     ]
                                # )
                                # test_cases.append(
                                #     [
                                #         b,
                                #         s,
                                #         n,
                                #         num_kv_heads,
                                #         h,
                                #         128,
                                #         True,
                                #         True,
                                #         True,
                                #         "context_attention_perf.txt",
                                #     ]
                                # )
                                # test_cases.append(
                                #     [
                                #         b,
                                #         s,
                                #         n,
                                #         num_kv_heads,
                                #         h,
                                #         0,
                                #         True,
                                #         False,
                                #         True,
                                #         "context_attention_perf.txt",
                                #     ]
                                # )
                                # test_cases.append(
                                #     [
                                #         b,
                                #         s,
                                #         n,
                                #         num_kv_heads,
                                #         h,
                                #         0,
                                #         True,
                                #         True,
                                #         True,
                                #         "context_attention_perf.txt",
                                #     ]
                                # )
                        # else:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                num_kv_heads,
                                h,
                                0,
                                False,
                                False,
                                True,
                                "context_attention_perf.txt",
                            ]
                        )
                        # if has_fp8:
                        #     test_cases.append(
                        #         [
                        #             b,
                        #             s,
                        #             n,
                        #             num_kv_heads,
                        #             h,
                        #             0,
                        #             True,
                        #             False,
                        #             True,
                        #             "context_attention_perf.txt",
                        #         ]
                        #     )
                        #     test_cases.append(
                        #         [
                        #             b,
                        #             s,
                        #             n,
                        #             num_kv_heads,
                        #             h,
                        #             0,
                        #             True,
                        #             True,
                        #             True,
                        #             "context_attention_perf.txt",
                        #         ]
                        #     )

    return test_cases


def get_generation_attention_test_cases():
    # has_fp8 = get_sm_version() > 86
    test_cases = []

    # generation
    b_list = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]
    # the i-th token to record. 1 for context phase. mapping to osl definition
    s_list = [
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    ]
    n_list = [4, 8, 12, 16, 24, 32, 40, 48, 64]
    n_list_xqa = [4, 8, 16, 32, 64, 96, 128]
    n_kv_list = [1, 2, 4, 8]
    head_dim = [64, 128]

    # MHA
    max_bsn = 8192 * 1024  # 2*1024*1024*1024/128/2 INT32MAX/128/2
    for n in sorted(n_list, reverse=True):
        b_s_dict = {}
        s_b_dict = {}
        for s in s_list:
            max_b = max_bsn // s // n  # b*s*n*byte <= max_bsn
            for b in b_list:
                if b > max_b:
                    break
                if s not in s_b_dict:
                    s_b_dict[s] = {b}
                else:
                    s_b_dict[s].add(b)
        for s, b_set in s_b_dict.items():
            if len(b_set) < 4:
                continue
            for b in b_set:
                if b not in b_s_dict:
                    b_s_dict[b] = {s - 1}
                b_s_dict[b].add(s - 1)
        for h in head_dim:
            for b, s_list_limited in b_s_dict.items():
                target_s_list = sorted(s_list_limited)
                if b >= 256:
                    target_s_list = target_s_list[:-1]
                # print(f'collecting MHA heads: {n} batchsize: {b}  steps: {s_list_limited}')
                # fp8 kv cache, fp8 context fmha, is_context_phase
                for s in target_s_list:
                    test_cases.append([b, s, n, n, h, 0, False, False, False, "generation_attention_perf.txt"])

                    if has_fp8:
                        test_cases.append([b, s, n, n, h, 0, True, False, False, "generation_attention_perf.txt"])
                        # currently, fp8 is not for generation compute
                        # test_cases.append(
                        #     [b, s, n, n, 128, True, True, False, "generation_attention_perf.txt"]
                        # )

    # XQA
    max_bsn = 8192 * 1024 * 2  # 2*1024*1024*1024/128/2
    for n in sorted(n_list_xqa, reverse=True): #4,...,128
        b_s_dict = {}
        s_b_dict = {}
        for s in s_list:
            max_b = max_bsn // s // n
            for b in b_list:
                if b > max_b:
                    break
                if s not in s_b_dict:
                    s_b_dict[s] = {b}
                else:
                    s_b_dict[s].add(b)
        for s, b_set in s_b_dict.items():
            if len(b_set) < 4:
                continue
            for b in b_set:
                if b not in b_s_dict:
                    b_s_dict[b] = {s - 1}
                b_s_dict[b].add(s - 1)
        for h in head_dim:
            for b, s_list_limited in b_s_dict.items():
                target_s_list = sorted(s_list_limited)
                if b >= 256:
                    target_s_list = target_s_list[:-1]
                for n_kv in n_kv_list:
                    if n_kv >= n:
                        continue

                    # fp8 kv cache, fp8 context fmha, is_context_phase
                    for s in target_s_list:
                        # if get_sm_version() >= 100:
                        #     # TLLM_CHECK_WITH_INFO((params.m_num_heads_q_per_kv < max_num_heads_q_per_kv_in_cta || params.m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta == 0), # noqa: E501
                        #     m_num_heads_q_per_kv = 1 if n_kv == 0 else n // n_kv
                        #     max_num_heads_q_per_kv_in_cta = 32
                        #     if (
                        #         m_num_heads_q_per_kv >= max_num_heads_q_per_kv_in_cta
                        #         and m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta != 0
                        #     ):
                        #         continue
                        if h == 64:
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    n_kv,
                                    h,
                                    128,
                                    False,
                                    False,
                                    False,
                                    "generation_attention_perf.txt",
                                ]
                            )
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    n_kv,
                                    h,
                                    0,
                                    False,
                                    False,
                                    False,
                                    "generation_attention_perf.txt",
                                ]
                            )
                            # if has_fp8:
                            #     test_cases.append(
                            #         [
                            #             b,
                            #             s,
                            #             n,
                            #             n_kv,
                            #             h,
                            #             128,
                            #             True,
                            #             False,
                            #             False,
                            #             "generation_attention_perf.txt",
                            #         ]
                            #     )
                            #     test_cases.append(
                            #         [
                            #             b,
                            #             s,
                            #             n,
                            #             n_kv,
                            #             h,
                            #             0,
                            #             True,
                            #             False,
                            #             False,
                            #             "generation_attention_perf.txt",
                            #         ]
                            #     )
                                # currently, fp8 is not for generation compute
                                # test_cases.append(
                                #     [
                                #         b,
                                #         s,
                                #         n,
                                #         n_kv,
                                #         128,
                                #         True,
                                #         True,
                                #         False,
                                #         "generation_attention_perf.txt",
                                #     ]
                                # )
                        else:
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    n_kv,
                                    h,
                                    0,
                                    False,
                                    False,
                                    False,
                                    "generation_attention_perf.txt",
                                ]
                            )
                            # if has_fp8:
                            #     test_cases.append(
                            #         [
                            #             b,
                            #             s,
                            #             n,
                            #             n_kv,
                            #             h,
                            #             0,
                            #             True,
                            #             False,
                            #             False,
                            #             "generation_attention_perf.txt",
                            #         ]
                            #     )
    return test_cases


if __name__ == "__main__":
    test_cases = get_context_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case)

    test_cases = get_generation_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case)
