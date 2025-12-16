# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

# import tensorrt_llm
import torch
import torch.nn as nn
import habana_frameworks.torch as htorch
# from tensorrt_llm._torch.modules.linear import Linear
# from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

import time
from helper import log_perf #get_sm_version, 
import subprocess
from packaging import version
import habana_frameworks.torch.utils.experimental as htexp


def get_habana_fw_version():
    output = subprocess.run(
        "pip list | grep habana-torch-plugin",
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return str(version.parse(output.stdout.split("\n")[0].split()[-1]))

def get_device_name():
    device_type = htexp._get_device_type()

    if device_type == htexp.synDeviceType.synDeviceGaudi:
        return "gaudi"
    elif device_type == htexp.synDeviceType.synDeviceGaudi2:
        return "gaudi2"
    elif device_type == htexp.synDeviceType.synDeviceGaudi3:
        return "gaudi3"

def get_gemm_test_cases():
    x_list = [
        1,
        2,
        4,
        8,
        # 16,
        # 32,
        # 48,
        # 64,
        # 80,
        # 96,
        # 128,
        # 160,
        # 192,
        # 256,
        # 384,
        # 512,
        # 768,
        # 1024,
        # 2048,
        # 4096,
        # 8192,
    ]
    nk_list = [
        32,
        # 64,
        # 128,
        # 256,
        # 512,
        # 768,
        # 1024,
        # 1536,
        # 2048,
        # 2560,
        # 3072,
        # 3584,
        # 4096,
        # 5120,
        # 6144,
        # 7168,
        # 8192,
        # 10240,
        # 12288,
    ]
    nk_list_ext = []#16384, 65536]  # for coverage and interp purpose
    gemm_list = ["bfloat16"]#, "fp8"]
    # if get_sm_version() > 86:
    #     gemm_list += ["fp8"]
    #     if get_sm_version() < 100:
    #         gemm_list += ["fp8_block"]
    # if get_sm_version() >= 100:
    #     gemm_list += ["nvfp4"]

    test_cases = []
    for gemm_type in gemm_list:
        # x_list_orig+add+ext  <==> nk_list+ext
        for x in sorted(x_list, reverse=True):
            for n in sorted(nk_list + nk_list_ext, reverse=True):
                for k in sorted(nk_list + nk_list_ext, reverse=True):
                    # if n * k == 65536 * 65536:
                    #     continue
                    # if (gemm_type == "nvfp4" or gemm_type == "fp8_block") and (n < 128 or k < 128):
                    #     continue
                    test_cases.append([gemm_type, x, n, k, "gemm_perf.txt"])

    return test_cases


def run_gemm(gemm_type, m, n, k, perf_filename, device="hpu",device_id=0):
    print("****************device id", device_id)
    device = torch.device(device)
    # torch.hpu.set_device(device)
    # torch.set_default_device(device)

    dtype = torch.bfloat16
    x = torch.randn((m, k), dtype=dtype).to(device)

    # if gemm_type == "fp8":
    #     qc = QuantConfig(quant_algo=QuantAlgo.FP8)
    # elif gemm_type == "fp8_block":
    #     group_size = 128
    #     qc = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=group_size)
    # elif gemm_type == "nvfp4":
    #     group_size = 128
    #     qc = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=group_size)
    # else:
    #     qc = None

    repeat_n = 5  # to reduce impact of L2 cache hit
    op_list = []
    for i in range(repeat_n):
        gemm = nn.Linear(
            in_features=k,
            out_features=n,
            bias=False
        ).to(dtype).to(device)

        # if gemm_type == "fp8":
        #     weights = {
        #         "weight": torch.randn((n, k), dtype=torch.bfloat16, device=torch.device(device)).to(
        #             dtype=torch.float8_e4m3fn
        #         ),
        #         "weight_scale": torch.randn(1, dtype=torch.float32, device=torch.device(device)),
        #     }
        # elif gemm_type == "fp8_block":
        #     weights = {
        #         "weight": torch.randn((n, k), dtype=torch.bfloat16, device=torch.device(device)).to(
        #             dtype=torch.float8_e4m3fn
        #         ),
        #         "weight_scale": torch.randn(
        #             (math.ceil(n / group_size), math.ceil(k / group_size)),
        #             dtype=torch.float32,
        #             device=torch.device(device),
        #         ),
        #     }
        # elif gemm_type == "nvfp4":
        #     # From trtllm test case
        #     x_sf_global = (448 * 6) / x.abs().max().float()
        #     w = torch.randn((n, k), dtype=torch.float16, device=torch.device(device))
        #     w_sf_global = (448 * 6) / w.abs().max().float()
        #     w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global, 16, False)
        #     w_sf_block_unswizzled = torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(w_sf_block.cpu().view(k, -1))
        #     weights = {
        #         "weight": w_fp4.cpu(),
        #         "weight_scale": w_sf_block_unswizzled.view(torch.float8_e4m3fn),
        #         "weight_scale_2": 1.0 / w_sf_global.cpu(),
        #         "input_scale": 1.0 / x_sf_global.cpu(),
        #     }
        # else:
        #     weights = {"weight": torch.randn((n, k), dtype=torch.bfloat16, device=torch.device(device))}

        # gemm.load_weights([weights])
        # gemm.to(torch.device(device))
        gemm(x)  # dry run to init
        op_list.append(gemm)

    num_warmups = 3
    num_runs = 6

    # capture
    # g = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(g):
    #     for op in op_list:
    #         op.forward(x)
    # warmup
    for op in op_list:
        for i in range(num_warmups):
            op.forward(x)

    start_time = time.time()
    for op in op_list:
        for i in range(num_runs):
            op.forward(x)
    torch.hpu.synchronize()
    latency = (time.time() - start_time) / num_runs / len(op_list)

    log_perf(
        item_list=[{"gemm_dtype": gemm_type, "m": m, "n": n, "k": k, "latency": latency}],
        framework="HPU",
        version=get_habana_fw_version(),
        device_name=get_device_name(),
        op_name="gemm",
        kernel_source="torch_compile",
        perf_filename=perf_filename,
    )

