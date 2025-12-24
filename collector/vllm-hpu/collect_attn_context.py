# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import time
import json
import math

import torch
from torch.profiler import schedule, profile, record_function, ProfilerActivity
from vllm.platforms import current_platform
from vllm.utils import is_torch_equal_or_newer
# from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.version import __version__ as vllm_version

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
import habana_frameworks.torch.utils.experimental as htexp

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

    device = torch.device(device)
    import habana_frameworks.torch.core as htcore

    dtype = torch.bfloat16
    model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
    block_size = 128
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

    from vllm_gaudi.extension.utils import ModuleFusedSDPA
    import vllm_gaudi.extension.kernels as kernels
    import habana_frameworks.torch as htorch

    HPUFusedSDPA = kernels.fsdpa()
    fsdpa_op = ModuleFusedSDPA(HPUFusedSDPA)

    test_ite = 6
    warm_up = 3
    fsdpa_op = htorch.hpu.wrap_in_hpu_graph(fsdpa_op)
    
    hidden_size = num_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim

    query = torch.randn(batch_size, input_len, num_heads, head_dim, dtype=torch.bfloat16, device="hpu")
    key = torch.randn(batch_size, input_len, num_heads, head_dim, dtype=torch.bfloat16, device="hpu")
    value = torch.randn(batch_size, input_len, num_heads, head_dim, dtype=torch.bfloat16, device="hpu")

    # Warmup
    warmup_time = time.time()
    for i in range(warm_up):
        fsdpa_op(
            query,
            key,
            value,
            None,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0/(head_dim**0.5),
            softmax_mode="fast",
            recompute_mode=True,
            valid_sequence_lengths=None
        )
    torch.hpu.synchronize()

    start_time = time.time()
    
    # Setup profiler activities - CPU and HPU
    activities = [ProfilerActivity.CPU, ProfilerActivity.HPU]
    # HPU operations are tracked through CPU profiler activity
    # as PyTorch HPU backend operations appear in CPU trace
    
    # with profile(
    #     activities=activities,
    #     schedule=schedule(wait=0, warmup=0, active=test_ite, repeat=1),
    #     with_stack=True,
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs')
    # ) as prof:
    for i in range(test_ite):
        fsdpa_op(
            query,
            key,
            value,
            None,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0/(head_dim**0.5),
            softmax_mode="fast",
            recompute_mode=True,
            valid_sequence_lengths=None
        )
        # profiler.step()
    torch.hpu.synchronize()

    latency = (time.time() - start_time) / test_ite * 1000

    isl = input_len
    step = 0
    op_name = "context_attention"

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


def get_context_attention_test_cases(if_unit_test=False):
    test_cases = []

    if not if_unit_test:
        b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256] #
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
        n_list = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
        n_kv_list = [0, 1, 2, 4, 8]
        # n_kv_list = [64]
    else:
        b_list = [1]
        s_list = [64]
        n_list = [4]
        n_kv_list = [0]

    kv_cache_dtype_list = [False]
    # if get_sm_version() > 86:
    #     kv_cache_dtype_list.append(True)

    # DEBUG
    # print(f"b_list: {b_list}, s_list: {s_list}, n_list: {n_list}, n_kv_list: {n_kv_list}")
    for n in sorted(n_list, reverse=True):
        for s in sorted(s_list, reverse=True):
            for b in sorted(b_list, reverse=True):
                for n_kv in n_kv_list:
                    if n_kv != 0 and (n_kv > n or n % n_kv != 0):
                        continue
                    num_kv_heads = n_kv if n_kv != 0 else n
                    # Only keep self-attention case
                    # if n != num_kv_heads:
                    #    continue
                    if num_kv_heads == n:
                        if b * s > 65536 or b > 128:
                            continue
                    else:
                        if b * s > 131072:
                            continue
                    if b * s * num_kv_heads * 128 * 2 >= 2147483647:
                        continue

                    for is_fp8_kv_cache in kv_cache_dtype_list:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                num_kv_heads,
                                128,
                                is_fp8_kv_cache,
                                True,
                                "context_attention_perf.txt",
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
