# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from argparse import ArgumentParser
import os
import torch
from packaging import version
import habana_frameworks.torch.utils.experimental as htexp
from helper import log_perf

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

def hccl_benchmark(dtype: str, hccl_op: str = "all_gather", test_range: str = "10,10000000,1000", num_gpus: int = 8):
    # nccl_test_bin = ""
    # if nccl_op == "all_gather":
    #     nccl_test_bin = "all_gather_perf"
    # elif nccl_op == "alltoall":
    #     nccl_test_bin = "alltoall_perf"
    # elif nccl_op == "reduce_scatter":
    #     nccl_test_bin = "reduce_scatter_perf"
    # elif nccl_op == "all_reduce":
    #     nccl_test_bin = "all_reduce_perf"
    # assert nccl_test_bin != ""
    test_name_map = {
        "all_gather": "all_gather",
        "alltoall": "all2all",
        "reduce_scatter": "reduce_scatter",
        "all_reduce": "all_reduce",
    }

    hccl_test = test_name_map[hccl_op]

    dtype_map = {
        "half": "bfloat16",
        "int8": "float8"
    }
    hccl_dtype = dtype_map[dtype]

    hccl_demo_script = "/root/npu-stack/hccl_demo/run_hccl_demo.py"

    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    size = min_size

    # major, minor, patch = torch.cuda.nccl.version()
    # nccl_version = f"{major}.{minor}.{patch}"
    nccl_version = "unknown"

    bytes_per_element = 2 if dtype == "half" else 1

    while size < max_size:
        inner_loop = 100 if size <= 16777216 else 60
    
        cmd_args = [
            "python3",
            hccl_demo_script,
            "--test", hccl_test,
            "--size", str(size),
            "--loop", str(inner_loop),
            "--nranks", str(num_gpus),
            "--ranks_per_node", str(num_gpus),
            "--node_id", "0",
            "--data_type", hccl_dtype,
            "--measure", "latency",
        ]
        env = os.environ.copy()
        if "HCCL_COMM_ID" not in env:
            env["HCCL_COMM_ID"] = "127.0.0.1:5555"

        result = subprocess.run(cmd_args, capture_output=True, text=True, env=env)
        print_lines = result.stdout.split("\n")
        for line in print_lines:
            # Look for "Host Latency" or "Device Latency" lines
            if "Device Latency" in line:
                # Extract the latency value (format: "Host Latency   : 0.451567 ms")
                parts = line.split(":")
                if len(parts) >= 2:
                    latency_str = parts[1].strip()
                    # Remove "ms" if present and extract the number
                    latency_str = latency_str.replace("ms", "").strip()
                    try:
                        latency = float(latency_str) #ms
                        break
                    except ValueError:
                        continue
        print(result)

        print(f"HCCL demo ({hccl_test})", f"{size=}, {latency=}")
        log_perf(
            item_list=[
                {
                    "nccl_dtype": dtype,
                    "num_gpus": num_gpus,
                    "message_size": size // bytes_per_element,
                    "latency": latency,
                }
            ],
            framework="HPU",
            version=get_habana_fw_version(),
            device_name=get_device_name(),
            op_name=hccl_op,
            kernel_source="HCCL",
            perf_filename="hccl_perf.txt",
        )

        size *= ratio


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hccl_op",
        "-HCCL",
        default="all_gather",
        choices=["all_gather", "alltoall", "reduce_scatter", "all_reduce"],
        help="HCCL OP: all_gather, alltoall, reduce_scatter, all_reduce",
    )
    parser.add_argument("--dtype", "-t", default="half", choices=["half", "int8"], help="HCCL OP data type")
    parser.add_argument(
        "--range",
        "-r",
        default="512,536870913,2",  # 512B to 512MB
        help="min_size,max_size,multiplicative_ratio",
    )
    parser.add_argument("--num_gpus", "-n", default=8, type=int)
    args = parser.parse_args()

    hccl_benchmark(args.dtype, args.hccl_op, args.range, args.num_gpus)
