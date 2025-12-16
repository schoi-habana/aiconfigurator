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

class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        # Add float versions for flashinfer
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0

class MockModelInput:
    """Mocks the ModelInputForGPU class which holds the slot_mapping."""
    def __init__(self, slot_mapping: torch.Tensor):
        self.slot_mapping = slot_mapping

class MockInputBuilder:
    """
    Mocks the ModelRunnerInputBuilder interface required by vLLM V1/0.6+ builders.
    Holds context required to build metadata.
    """
    def __init__(self, vllm_config, kv_cache_spec, layer_names, device, slot_mapping):
        self.vllm_config = vllm_config
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.device = device
        
        self.model_input = MockModelInput(slot_mapping)

        # Attributes often accessed by the builder
        self.block_size = vllm_config.cache_config.block_size
        self.sliding_window = None 
        self.num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
        
        # Mocking runner/scheduler context if needed
        self.decode_log_interval = 10 

# https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends
# support MHA GQA MQA fp16 tensor and float16/fp8 kv cache


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
    impl_cls = backend_cls.get_impl_cls()

    if is_context_phase:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size, #[16, 16]
            query_lens=[input_len] * batch_size, #[16, 16]
        )
    else:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size,
            query_lens=[1] * batch_size,
        )

    
    current_platform.seed_everything(42)
    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=8192,
        max_num_seqs=batch_size,
    )
    import vllm_gaudi.extension.environment as environment
    environment.set_vllm_config(vllm_config) #hpu specific

    # kv_cache_spec = create_standard_kv_cache_spec(vllm_config, use_fp8_kv_cache)

    # Generate data and compute SDPA reference output
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence
        q = torch.randn(q_len, num_heads, head_dim, dtype=dtype, device=device) #16,2,128
        k_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Inputs for vLLM backends are just the new tokens
        # all_q_vllm.append(q) #(16,2,128)
        # all_k_vllm.append(k_full[context_len:])
        # all_v_vllm.append(v_full[context_len:])
        q = q.view(q_len, num_heads * head_dim)
        k_full = k_full.view(s_len, num_kv_heads * head_dim)
        v_full = v_full.view(s_len, num_kv_heads * head_dim) 

        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        # Contextual K/V data used to populate the paged cache
        k_contexts.append(k_full[:context_len].view(context_len, num_kv_heads, head_dim))
        v_contexts.append(v_full[:context_len].view(context_len, num_kv_heads, head_dim))

    query_vllm = torch.stack(all_q_vllm, dim=0) #32,2,128
    key_vllm = torch.stack(all_k_vllm, dim=0)
    value_vllm = torch.stack(all_v_vllm, dim=0)

    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)
    metadata_cls = backend_cls.get_metadata_cls() 
    
    kv_cache_spec = vllm_config 
    layer_names = ["placeholder"]
    total_tokens = sum(batch_spec.seq_lens)
    total_new_tokens = sum(batch_spec.query_lens)

    slot_mapping = torch.arange(total_new_tokens, dtype=torch.long, device=device)

    #  For HPU backends, we need to construct the metadata properly
    # Check if it's HPUAttentionMetadataV1 which has factory methods
    if hasattr(metadata_cls, 'make_prefill_metadata') and is_context_phase:
        # Use the factory method for prefill
        seq_lens_tensor = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device) #(bs, sl)
        context_lens = [batch_spec.seq_lens[i] - batch_spec.query_lens[i] for i in range(batch_size)] #0s
        context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, device=device)#0s
        
        attn_metadata = metadata_cls.make_prefill_metadata(
            attn_bias=None,
            block_list=None,
            context_lens_tensor=context_lens_tensor,
            seq_lens_tensor=seq_lens_tensor,
            slot_mapping=slot_mapping,
            block_size=block_size,
            query_start_loc=None,
        )
    elif hasattr(metadata_cls, 'make_decode_metadata') and not is_context_phase:
        # Use the factory method for decode
        num_active_blocks = batch_size
        block_list = torch.arange(num_active_blocks, dtype=torch.long, device=device)#.unsqueeze(1)
        # num_blocks = vllm_config.cache_config.num_gpu_blocks
        block_mapping = torch.zeros(num_active_blocks, batch_size, dtype=dtype, device=device)
        block_mapping[torch.arange(num_active_blocks), torch.arange(batch_size)] = 1.0
        # for i in range(batch_size):
        #     # Map each batch item to its corresponding block (wrap around if needed)
        #     block_idx = i % num_blocks
        #     block_mapping[block_idx, i] = 1.0
        block_groups = torch.arange(num_active_blocks, dtype=torch.long, device=device) % batch_size
        block_usage = torch.full((num_active_blocks,), block_size, dtype=dtype, device=device)

        attn_metadata = metadata_cls.make_decode_metadata(
            block_list=block_list,
            block_usage=block_usage,#None,
            block_groups=block_groups,#None,
            input_positions=torch.arange(total_new_tokens, dtype=torch.long, device=device),
            slot_mapping=slot_mapping,
            block_size=block_size,
            window_block_list=None,
            window_block_usage=None,
            window_block_groups=None,
            query_start_loc=None,
        )
        if hasattr(attn_metadata, 'block_mapping'):
            attn_metadata.block_mapping = block_mapping
        if hasattr(attn_metadata, 'attn_bias') and attn_metadata.attn_bias is None:
            # Create a zero bias tensor: shape will be reshaped to (batch_size, 1, 1, block_size)
            attn_bias_shape = (num_active_blocks, block_size)
            attn_metadata.attn_bias = torch.zeros(attn_bias_shape, dtype=dtype, device=device)

    num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
    
    # Note: Check if backend provides a specific cache creation utility
    # For now, generic allocation:
    key_cache = torch.zeros(
        (num_gpu_blocks*block_size, num_kv_heads, head_dim), 
        dtype=dtype, device=device
    )
    value_cache = torch.zeros(
        (num_gpu_blocks*block_size, num_kv_heads, head_dim), 
        dtype=dtype, device=device
    )
    kv_cache = (key_cache, value_cache)
    # If FP8 is requested:
    # if use_fp8_kv_cache:
    #     kv_cache = kv_cache.to(torch.float8_e4m3fn) # Gaudi supports FP8

    
    impl_cls = backend_cls.get_impl_cls()
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / (head_dim**0.5),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else "auto"
    )

    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query_vllm)

    # Instantiate implementation
    sliding_window = vllm_config.model_config.get_sliding_window()
    scale = 1.0 / (head_dim**0.5)

    test_ite = 6
    warm_up = 3
    
    def run():
        # htcore.mark_step()
        impl.forward(
            mock_layer,
            query_vllm,
            key_vllm,
            value_vllm,
            kv_cache,
            attn_metadata,
            output=output,
        )
        # htcore.mark_step()

    # Warmup
    for i in range(warm_up):
        run()
    torch.hpu.synchronize()

    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    

    def trace_handler(p):
        filename = get_file_name()
        p.export_chrome_trace(filename)
    
    def calculate_gemm_pipeline_span(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        events = data.get('traceEvents', [])

        # 1. Filter specifically for GEMM kernels
        # We look for cat="kernel" and name="gemm"
        gemm_kernels = [
            e for e in events 
            if e.get('cat') == 'kernel' and e.get('name') == 'gemm'
        ]

        if not gemm_kernels:
            print("No 'gemm' kernels found in the trace.")
            return

        # 2. Sort events by timestamp (ts) to find the chronological order
        sorted_kernels = sorted(gemm_kernels, key=lambda x: x['ts'])

        # 3. Identify First and Last
        first_kernel = sorted_kernels[0]
        last_kernel = sorted_kernels[-1]

        # 4. Calculate Timestamps
        start_time = first_kernel['ts']
        
        # End time is the start of the last kernel + its duration
        end_time = last_kernel['ts'] + last_kernel['dur']
        
        # 5. Calculate Total Span
        pipeline_span_us = end_time - start_time
        pipeline_span_ms = pipeline_span_us / 1000.0

        print(f"--- Pipeline Analysis for 'gemm' ---")
        print(f"Count: {len(gemm_kernels)} ops")
        print(f"Start (First Op): {start_time}")
        print(f"End   (Last Op):  {end_time}")
        print(f"Total Pipeline Span: {pipeline_span_ms:.4f} ms")
        os.remove(json_path)
        return pipeline_span_ms

    # torch.cuda.synchronize()
    # start_event.record()
    # start_time = time.perf_counter()
    import habana_frameworks.torch.core as htcore
    pid = os.getpid()
    timestamp = int(time.time())
    filename = f"trace_pid{pid}_ts{timestamp}.json"
    with profile(schedule=schedule(wait=0,warmup=0,active=test_ite,repeat=1),
        activities=[ProfilerActivity.CPU, ProfilerActivity.HPU],
        on_trace_ready=lambda p:p.export_chrome_trace(filename),
        record_shapes=False, with_stack=True) as prof:
        for i in range(test_ite):
            with record_function("run"):
                run()
            prof.step()
    latency = calculate_gemm_pipeline_span(filename)
    print("*******latency", latency, "ms")

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
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
