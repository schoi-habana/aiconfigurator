# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import time
import json

import torch
from torch.profiler import schedule, profile, record_function, ProfilerActivity
from vllm.platforms import current_platform
from vllm.utils import is_torch_equal_or_newer
# from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.version import __version__ as vllm_version
from vllm_gaudi.extension.runtime import get_config

from collector.vllm.collector_vllm_utils import (
    BatchSpec,
    # _Backend,
    create_and_prepopulate_kv_cache,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    get_attention_backend,
    resolve_obj_by_qualname,
)
# from vllm.utils import resolve_obj_by_qualname
from helper import log_perf#get_sm_version, 

from vllm_gaudi.utils import async_h2d_copy
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.experimental as htexp

import vllm_gaudi.extension.ops as ops
from vllm_gaudi.extension.ops import flat_pa

from vllm_gaudi.extension.utils import Matmul, VLLMKVCache
def get_device_name():
    device_type = htexp._get_device_type()

    if device_type == htexp.synDeviceType.synDeviceGaudi:
        return "gaudi"
    elif device_type == htexp.synDeviceType.synDeviceGaudi2:
        return "gaudi2"
    elif device_type == htexp.synDeviceType.synDeviceGaudi3:
        return "gaudi3"


def run_attention_torch(
    batch_size,
    input_len,
    num_heads,
    num_kv_heads,  # keep same as num_heads for MHA
    head_dim,
    use_fp8_kv_cache,
    is_context_phase,
    perf_filename,
    device="hpu",
    device_id=0,
):
    model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
    block_size = 128

    dtype = torch.bfloat16
    backend_class_name = current_platform.get_attn_backend_cls(
        None,
        head_dim,
        dtype,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
        block_size=block_size,
        use_v1=True,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
    )

    backend_cls = resolve_obj_by_qualname(backend_class_name)
    
    backend_name = backend_cls.get_name()

    current_platform.seed_everything(42)
    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=input_len,
        block_size=block_size,
        num_gpu_blocks=8192,
        max_num_seqs=batch_size,
    )
    import vllm_gaudi.extension.environment as environment
    environment.set_vllm_config(vllm_config) #hpu specific

    query_len = 1

    # device = torch.device(device)
    query = torch.randn(batch_size, query_len, num_heads * head_dim, dtype=torch.bfloat16, device=device).contiguous()
    new_key = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    new_value = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    context_len = input_len - query_len
    key_cache = torch.randn(8192 * block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device).contiguous() # num_blocks, block_size=128
    value_cache = torch.randn(8192 * block_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device).contiguous()

    slot_mapping = torch.arange(batch_size, dtype=torch.long, device=device)
    

    num_active_blocks = max(1, (context_len + block_size - 1) // block_size) if context_len > 0 else 1
    block_list = torch.arange(num_active_blocks, dtype=torch.int32, device=device)
    block_bias = torch.zeros((num_active_blocks,), dtype=torch.bfloat16, device=device)
    block_groups = torch.zeros((num_active_blocks,), dtype=torch.long, device=device)
    block_mapping = torch.zeros((num_active_blocks, batch_size), dtype=torch.bfloat16, device=device)
    for i in range(batch_size):
        block_idx = i % num_active_blocks
        block_mapping[block_idx, i] = 1.0
    block_mapping = block_mapping.contiguous()
    scale = 1.0 / (head_dim ** 0.5)

    kv_cache_obj = VLLMKVCache()
    kv_cache_obj.forward(input=new_key, cache=key_cache, slot_mapping=slot_mapping)

    matmul_qk_op = Matmul()
    matmul_av_op = Matmul()
    batch2block_matmul = Matmul()
    block2batch_matmul = Matmul()

    def run_flat_pa(query, key_cache, value_cache, block_list, block_mapping, 
                block_bias, block_groups, block_size, scale, 
                matmul_qk_op, matmul_av_op, batch2block_matmul, block2batch_matmul,
                keys_fetch_func, values_fetch_func):

        return flat_pa(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_list=block_list,
            block_mapping=block_mapping,
            block_bias=block_bias,
            block_groups=block_groups,
            block_size=block_size,
            scale=scale,
            matmul_qk_op=matmul_qk_op,
            position_bias=None,
            matmul_av_op=matmul_av_op,
            batch2block_matmul_op=batch2block_matmul,
            block2batch_matmul_op=block2batch_matmul,
            keys_fetch_func=keys_fetch_func,
            values_fetch_func=values_fetch_func,
            k_scales=None,
            v_scales=None
        )

    test_ite = 6
    warm_up = 3
    
    import habana_frameworks.torch as htorch
    
    s = htcore.hpu.Stream()
    with htcore.hpu.stream(s):
        g = htorch.hpu.HPUGraph()

        g.capture_begin()
        run_flat_pa(query, key_cache, value_cache, block_list, block_mapping,
                block_bias, block_groups, block_size, scale,
                matmul_qk_op, matmul_av_op, batch2block_matmul, block2batch_matmul,
                kv_cache_obj.fetch_from_cache, kv_cache_obj.fetch_from_cache)
        g.capture_end()
    
    torch.hpu.synchronize()
    start_time = time.time()
    for _ in range(test_ite):
        g.replay()
        torch.hpu.synchronize()
    latency = (time.time() - start_time) / test_ite * 1000

    isl = 1
    step = input_len
    op_name = "generation_attention"

    kv_cache_dtype_str = "bfloat16" if not use_fp8_kv_cache else "fp8"
    dtype_str = "bfloat16"
    kernel_source = f"vllm_{backend_name}".lower()

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "beam_width": 1,
                "attn_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "step": step,
                "latency": latency,
            }
        ],
        framework="VLLM-HPU",
        version=vllm_version,
        device_name=get_device_name(),
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
    )



def get_generation_attention_test_cases():
    test_cases = []

    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # b_list_xqa = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    n_list = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
    # n_list_xqa = [4,8,16,32,64,128]
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
    n_kv_list = [1, 2, 4, 8]

    kv_cache_dtype_list = [False]
    # if get_sm_version() > 86:
    #     kv_cache_dtype_list.append(True)

    max_bsn = 8192 * 1024
    for n in sorted(n_list, reverse=True):
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
        for b, s_list_limited in b_s_dict.items():
            target_s_list = sorted(s_list_limited)
            if b >= 256:
                target_s_list = target_s_list[:-1]
            for n_kv in n_kv_list:
                if n_kv > n or n % n_kv != 0:
                    continue
                for s in target_s_list:
                    for is_fp8_kv_cache in kv_cache_dtype_list:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                n_kv,
                                128,
                                is_fp8_kv_cache,
                                False,
                                "generation_attention_perf.txt",
                            ]
                        )
    return test_cases


if __name__ == "__main__":
    test_cases = get_context_attention_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running context attention test case: {test_case}")
        run_attention_torch(*test_case)

    test_cases = get_generation_attention_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running generation attention test case: {test_case}")
        run_attention_torch(*test_case)
