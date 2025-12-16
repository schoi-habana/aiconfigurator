import math

best_compute_time = float('inf')
best_hbm_time = float('inf')
best_c2c_time = float('inf')
gemm_rows = 8192
gemm_cd = 128
gemm_cols = 4096
gemm_num_gemms = 64
num_dcores = 4
mme_per_dcore = 2
input_bytes = 2
output_bytes = 2
port_bytes = 128
opA_placement = 'noAlloc'
opB_placement = 'noAlloc'
opA_alloc_policy = 'allocD'
opB_alloc_policy = 'allocH'
opC_tag_alloc_policy = 'allocD'
opC_alloc_policy = 'noAlloc'
dcore_capacity_mb = 24
num_of_mme_output_ports = 2
mme_freq_ghz = 1.6
single_r2c_bw = 1600
single_c2r_bw = 3200
single_c2c_bw = 400
cacheline_bytes = 256
mme_geo_height = 256
mme_geo_width = 256
single_hbm_bw = 800

def g3_single_mme_sb_reuse(rows, cd, cols, input_bytes):
    # General calculations:
    cacheline_elems = cacheline_bytes / input_bytes
    cd_cachelines = math.ceil(cd/cacheline_elems)
    is_cd_fully_fit = 1 if cd_cachelines <= 25 else 0
    max_cds_fit_sb = math.floor(25/cd_cachelines)
    opA_num_of_cachelines = math.ceil(cd/cacheline_elems) * rows
    opB_num_of_cachelines = math.ceil(cols / cacheline_elems) * cd
    num_of_geos_fit_to_the_height = math.ceil(rows / mme_geo_height)
    num_of_geos_fit_to_the_width = math.ceil(cols / mme_geo_width)

    # Fully-Fit Calculations:
    max_cds_fit_sb_a = min(max_cds_fit_sb, num_of_geos_fit_to_the_height)
    max_cds_fit_sb_b = min(max_cds_fit_sb, num_of_geos_fit_to_the_width)
    num_of_reps_opA = math.ceil(max_cds_fit_sb_a / num_of_geos_fit_to_the_height)
    num_of_reps_opB = math.ceil(max_cds_fit_sb_b / num_of_geos_fit_to_the_width)
    a_reuse_num_of_read_cachlines = opA_num_of_cachelines + opB_num_of_cachelines * num_of_reps_opA
    b_reuse_num_of_read_cachlines = opB_num_of_cachelines + opA_num_of_cachelines * num_of_reps_opB
    fully_fit_num_of_read_cachelines = min(a_reuse_num_of_read_cachlines, b_reuse_num_of_read_cachlines)
    fully_fit_preferred_strategy = "A-Reuse" if fully_fit_num_of_read_cachelines == a_reuse_num_of_read_cachlines else "B-Reuse"
    fully_fit_num_of_a_rereads = 1 if fully_fit_preferred_strategy == "A-Reuse" else num_of_reps_opB
    fully_fit_num_of_b_rereads = num_of_reps_opA if fully_fit_preferred_strategy == "A-Reuse" else 1

    # Non Fully-Fit Calculations:
    height_num_of_read_cachelines = opA_num_of_cachelines * math.ceil(num_of_geos_fit_to_the_width/1) + opB_num_of_cachelines * math.ceil(num_of_geos_fit_to_the_height/4)
    sym_num_of_read_cachelines = opA_num_of_cachelines * math.ceil(num_of_geos_fit_to_the_width/2) + opB_num_of_cachelines * math.ceil(num_of_geos_fit_to_the_height/2)
    width_num_of_read_cachelines = opA_num_of_cachelines * math.ceil(num_of_geos_fit_to_the_width/4) + opB_num_of_cachelines * math.ceil(num_of_geos_fit_to_the_height/1)
    non_fully_fit_num_of_read_cachelines = min(height_num_of_read_cachelines, sym_num_of_read_cachelines, width_num_of_read_cachelines)
    if non_fully_fit_num_of_read_cachelines == height_num_of_read_cachelines:
        # non_fully_fit_preferred_strategy = "Height"
        non_fully_fit_num_of_a_rereads = math.ceil(num_of_geos_fit_to_the_width / 1)
        non_fully_fit_num_of_b_rereads = math.ceil(num_of_geos_fit_to_the_height / 4)
    elif non_fully_fit_num_of_read_cachelines == sym_num_of_read_cachelines:
        # non_fully_fit_preferred_strategy = "Sym"
        non_fully_fit_num_of_a_rereads = math.ceil(num_of_geos_fit_to_the_width / 2)
        non_fully_fit_num_of_b_rereads = math.ceil(num_of_geos_fit_to_the_height / 2)
    else:
        # non_fully_fit_preferred_strategy = "Width"
        non_fully_fit_num_of_a_rereads = math.ceil(num_of_geos_fit_to_the_width / 4)
        non_fully_fit_num_of_b_rereads = math.ceil(num_of_geos_fit_to_the_height / 1)

    # Results:
    # is_fully_fit = is_cd_fully_fit
    # num_of_read_cachelines = fully_fit_num_of_read_cachelines if is_cd_fully_fit else non_fully_fit_num_of_read_cachelines
    # preferred_strategy = fully_fit_preferred_strategy if is_cd_fully_fit else non_fully_fit_preferred_strategy
    num_of_a_rereads = fully_fit_num_of_a_rereads if is_cd_fully_fit else non_fully_fit_num_of_a_rereads
    num_of_b_rereads = fully_fit_num_of_b_rereads if is_cd_fully_fit else non_fully_fit_num_of_b_rereads
    return num_of_a_rereads, num_of_b_rereads

MAX_ACTIVATIONS = 23
PMU_MACS_BUDGET_DICT = {"fp8": {0.178: 133120},
                        "bf16": {0.089: 68608},
                        "fp16": {0.355: 262144},
                        "fp32": {0.352: 262144},
                        "tf32": {0.298: 229376}}
MAX_MACS_BUDGET_FOR_SINGLE_EU = 32768 # 128 * 256
def get_act_pmu_perc(single_mme_rows, single_mme_cols):
    pmu = round(0.8, 3)
    # pmu = round(userArgs.pmu, 3)
    if (pmu in PMU_MACS_BUDGET_DICT["bf16"].keys()):
        mac_budget = PMU_MACS_BUDGET_DICT["bf16"][pmu]
        rows = min(math.ceil(single_mme_rows / 2), mme_geo_height) # 2 EUs in each MME
        cols = min(single_mme_cols, mme_geo_width)
        cur_mac_budget = rows * cols
        device_pmu = 0.8
        achieved_activations = min(math.floor(device_pmu*MAX_ACTIVATIONS), min(MAX_ACTIVATIONS, math.floor(mac_budget/cur_mac_budget)))
        util_perc = (achieved_activations / MAX_ACTIVATIONS) * cur_mac_budget / MAX_MACS_BUDGET_FOR_SINGLE_EU
    else:
        # util_perc = userArgs.pmu
        util_perc = 0.8
    return util_perc

for i in range(0, 4):
    for j in range(0, 4):
        if i == 1 or j == 1:  # Turning off CD perforation and CD split - probably won't do
            continue
        input_array = [gemm_rows, gemm_cd, gemm_cols, gemm_num_gemms]
        input_array[i] = math.ceil(input_array[i] / num_dcores)
        single_dcore_rows, single_dcore_cd, single_dcore_cols, single_dcore_num_of_gemms = input_array #1,128,128,1
        input_array[j] = math.ceil(input_array[j] / mme_per_dcore)
        single_mme_rows, single_mme_cd, single_mme_cols, single_mme_num_of_gemms = input_array #1,128,128,1
        bgemm_mode = 0

        # -------------- Capacity calculations --------------:
        # Capacity calculations - Device Level:
        opA_bytes = math.ceil(gemm_cd * input_bytes / port_bytes) * port_bytes * gemm_rows * gemm_num_gemms
        opB_bytes = math.ceil(gemm_cols * input_bytes / port_bytes) * port_bytes * gemm_cd * gemm_num_gemms
        opC_bytes = math.ceil(gemm_cols * output_bytes / port_bytes) * port_bytes * gemm_rows * gemm_num_gemms
        # ----- Capacity calculations - DCORE Level -----:
        # GEMM size:
        single_dcore_opA_gemm_size = math.ceil(
            single_dcore_cd * input_bytes / port_bytes) * port_bytes * single_dcore_rows * single_dcore_num_of_gemms
        single_dcore_opB_gemm_size = math.ceil(
            single_dcore_cols * input_bytes / port_bytes) * port_bytes * single_dcore_cd * single_dcore_num_of_gemms
        single_dcore_opC_gemm_size = opC_bytes / num_dcores
        # Capacity in cache [bytes]:
        single_dcore_opA_bytes = opA_bytes / num_dcores * {"noAlloc": 0, "allocH": 1, "allocD": 1,
                                                                "allocDH": num_dcores if i == 2 else 1}.get(
            opA_alloc_policy, 2)
        single_dcore_opB_bytes = opB_bytes / num_dcores * {"noAlloc": 0, "allocH": 1, "allocD": 1,
                                                                "allocDH": num_dcores if i == 0 else 1}.get(
            opB_alloc_policy, 2)
        single_dcore_opC_tag_bytes = single_dcore_opC_gemm_size * (4 / output_bytes) * (
                    gemm_cd / single_mme_cd) if single_mme_cd < gemm_cd else 0
        single_dcore_opC_bytes = 0 if opC_alloc_policy == "noAlloc" else opC_bytes / num_dcores
        # Capacity in cache [%]:
        capacity_in_cache_percentage = ((single_dcore_opA_bytes + single_dcore_opB_bytes +
                                            single_dcore_opC_tag_bytes + single_dcore_opC_bytes) / 2 ** 20) / dcore_capacity_mb
        part_up_to_100_percent = min(1, capacity_in_cache_percentage)
        part_above_100_percent = max(0, capacity_in_cache_percentage - 1)
        # Miss Rate [%]:
        miss_rate_percentage = max(0, min(1, part_above_100_percent + 23.456 * (
                    part_up_to_100_percent ** 6) - 80.817 * (part_up_to_100_percent ** 5) +
                                            109.18 * (part_up_to_100_percent ** 4) - 73.506 * (
                                                        part_up_to_100_percent ** 3) +
                                            26.186 * (part_up_to_100_percent ** 2) - 4.6473 * (
                                                        part_up_to_100_percent ** 1) + 0.3194))
        # Chosen geo forced this rereads:
        num_of_a_rereads, num_of_b_rereads = g3_single_mme_sb_reuse(single_mme_rows, single_mme_cd,
                                                                    single_mme_cols, input_bytes)

        # -------------- Compute calculations --------------:
        num_of_geos = math.ceil(single_mme_rows/mme_geo_height) * math.ceil(single_mme_cols/mme_geo_width)
        eu_output_bytes = 4 if single_mme_cd < gemm_cd else output_bytes
        mme_write_cycles = min(single_mme_rows, mme_geo_height)*math.ceil((min(single_mme_cols, mme_geo_width)*eu_output_bytes/port_bytes)/num_of_mme_output_ports)
        mme_write_cycles *= 1/2 if bgemm_mode else 1
        single_geo_mme_cycles = max(mme_write_cycles, single_mme_cd * 8 if input_bytes == 4 else single_mme_cd)
        mme_cycles = single_geo_mme_cycles * num_of_geos * single_mme_num_of_gemms
        compute_time = mme_cycles/(mme_freq_ghz*1000)
        if single_mme_cd < gemm_cd:
            reduce_add_opC_tag_read_bytes = single_dcore_opC_tag_bytes
            reduce_add_opC_tag_write_bytes = single_dcore_opC_bytes
            reduce_add_time = max((reduce_add_opC_tag_read_bytes/2**30)/single_c2r_bw, (reduce_add_opC_tag_write_bytes/2**30)/single_r2c_bw) * 10**6
            compute_time += reduce_add_time
        total_tflops = gemm_rows * gemm_cd * gemm_cols * gemm_num_gemms * 2 / 10**12 / (compute_time/10**6)
        pmu_act_perc = get_act_pmu_perc(single_mme_rows, single_mme_cols)
        # pmu_slowdown_factor = max(1, (total_tflops/deviceArgs.max_mme_tflops)/deviceArgs.pmu)
        max_mme_tflops = mme_geo_height * mme_geo_width * 2 * 8 * 1.6 * 10**9 / 10**12
        pmu_slowdown_factor = max(1, (total_tflops/max_mme_tflops)/pmu_act_perc)
        compute_time *= pmu_slowdown_factor

        # -------------- BW Calculations --------------:
        # Single C2R:
        # c2r_opA_read_bytes = single_dcore_opA_gemm_size * num_of_a_rereads * (2 if j == 2 else 1)
        # c2r_opB_read_bytes = single_dcore_opB_gemm_size * num_of_b_rereads * (2 if j == 0 else 1)
        # c2r_opC_tag_read_bytes = single_dcore_opC_tag_bytes
        # c2r_bytes = c2r_opA_read_bytes + c2r_opB_read_bytes + c2r_opC_tag_read_bytes
        # c2r_time = (c2r_bytes/2**30) / deviceArgs.single_c2r_bw * 10**6

        # Single R2C:
        # r2c_opC_tag_write_bytes = single_dcore_opC_tag_bytes
        # r2c_opC_write_bytes = single_dcore_opC_gemm_size
        # r2c_bytes = r2c_opC_tag_write_bytes + r2c_opC_write_bytes
        # r2c_time = (r2c_bytes/2**30) / deviceArgs.single_r2c_bw * 10**6

        # Single C2C:
        c2c_opA_read_bytes = (single_dcore_opA_gemm_size/num_dcores)
        if opA_alloc_policy == "noAlloc" or opA_alloc_policy == "allocH":
            c2c_opA_read_bytes *= num_of_a_rereads * 2 if j == 2 else 1
        else:  # opA_alloc_policy == "allocD" or opA_alloc_policy == "allocDH"
            if opA_placement == "noAlloc":
                c2c_opA_read_bytes *= 1 + (num_of_a_rereads * (2 if j == 2 else 1) - 1) * miss_rate_percentage
            else:
                c2c_opA_read_bytes *= (num_of_a_rereads * (2 if j == 2 else 1) - 1) * miss_rate_percentage
                if i == 0:
                    c2c_opA_read_bytes *= 10  # ping pong penalty

        c2c_opB_read_bytes = (single_dcore_opB_gemm_size/num_dcores)
        if opB_alloc_policy == "noAlloc" or opB_alloc_policy == "allocH":
            c2c_opB_read_bytes *= num_of_b_rereads * 2 if j == 0 else 1
        else:  # opB_alloc_policy == "allocD" or opB_alloc_policy == "allocDH"
            if opB_placement == "noAlloc":
                c2c_opB_read_bytes *= 1 + (num_of_b_rereads * (2 if j == 0 else 1) - 1) * miss_rate_percentage
            else:
                c2c_opB_read_bytes *= (num_of_b_rereads * (2 if j == 0 else 1) - 1) * miss_rate_percentage
                if i == 0:
                    c2c_opB_read_bytes *= 10  # ping pong penalty

        c2c_opC_tag_read_bytes = single_dcore_opC_tag_bytes/num_dcores * \
                                    {"noAlloc": 1, "allocH": 1, "allocD": miss_rate_percentage}.get(opC_tag_alloc_policy, 2)
        c2c_opC_tag_write_bytes = single_dcore_opC_tag_bytes/num_dcores * \
                                    {"noAlloc": 1, "allocH": 1, "allocD": 1 if i==1 else miss_rate_percentage}.get(opC_tag_alloc_policy, 2)
        c2c_opC_write_bytes = single_dcore_opC_gemm_size/num_dcores * \
                                {"noAlloc": 1, "allocH": 1, "allocD": 0, "allocDH": 0}.get(opC_alloc_policy, 2)
        c2c_bytes = c2c_opA_read_bytes + c2c_opB_read_bytes + c2c_opC_tag_read_bytes + c2c_opC_tag_write_bytes + c2c_opC_write_bytes
        c2c_time = (c2c_bytes/2**30) / single_c2c_bw * 10**6

        # Single HBM:
        if opA_placement == "noAlloc":
            if opA_alloc_policy == "noAlloc":
                hbm_opA_read_bytes = single_dcore_opA_gemm_size * num_of_a_rereads * (2 if j == 2 else 1)
            else:
                hbm_opA_read_bytes = opA_bytes / num_dcores + (single_dcore_opA_gemm_size * num_of_a_rereads * (2 if j == 2 else 1) - (opA_bytes/num_dcores)) * miss_rate_percentage
        else:
            hbm_opA_read_bytes = single_dcore_opA_gemm_size * num_of_a_rereads * (2 if j == 2 else 1) * miss_rate_percentage * 0

        if opB_placement == "noAlloc":
            if opB_alloc_policy == "noAlloc":
                hbm_opB_read_bytes = single_dcore_opB_gemm_size * num_of_b_rereads * (2 if j == 0 else 1)
            else:
                hbm_opB_read_bytes = opB_bytes / num_dcores + (single_dcore_opB_gemm_size * num_of_b_rereads * (2 if j == 0 else 1) - (opB_bytes/num_dcores)) * miss_rate_percentage
        else:
            hbm_opB_read_bytes = single_dcore_opB_gemm_size * num_of_b_rereads * (2 if j == 0 else 1) * miss_rate_percentage * 0
        hbm_opC_tag_read_bytes = single_dcore_opC_tag_bytes / num_dcores * \
                                    {"noAlloc": 1, "allocH": miss_rate_percentage, "allocD": miss_rate_percentage}.get(opC_tag_alloc_policy, 2)
        hbm_opC_tag_write_bytes = hbm_opC_tag_read_bytes
        hbm_opC_write_bytes = single_dcore_opC_gemm_size * 1 if opC_alloc_policy == "noAlloc" else 0
        hbm_bytes = hbm_opA_read_bytes + hbm_opB_read_bytes + hbm_opC_tag_read_bytes + hbm_opC_tag_write_bytes + hbm_opC_write_bytes
        hbm_time = (hbm_bytes/2**30) / single_hbm_bw * 10**6

        if compute_time < best_compute_time or (compute_time == best_compute_time and hbm_time < best_hbm_time)\
                or (compute_time == best_compute_time and hbm_time <= 1.05 * best_hbm_time and c2c_time < 0.8 * best_c2c_time):
            best_compute_time = compute_time
            best_hbm_time = hbm_time
            best_c2c_time = c2c_time
total_time = max(best_compute_time, best_hbm_time, best_c2c_time)
print(total_time, "us") # nvidia number is in ms
