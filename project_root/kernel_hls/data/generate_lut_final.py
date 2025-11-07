import numpy as np
import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 0. 全局配置 (无变化)
# ==============================================================================
NUM_SEGMENTS = 128
HW_EXP_FIELD_START_RSQRT = 64
HW_EXP_FIELD_START_RECIPROCAL = 64
EXP_RANGE = [-16.0, 0.0]
ERF_RANGE = [0.0, 4.0]

# --- 函数及其解析导数 (无变化) ---
def exp_func(x): return np.exp(x)
def erf_func(x): return torch.erf(torch.tensor(x, dtype=torch.float32)).numpy()
def rsqrt_func(x): return 1.0 / np.sqrt(x)
def reciprocal_func(x): return 1.0 / x

def rsqrt_deriv(x): return -0.5 * np.power(x, -1.5)
def reciprocal_deriv(x): return -1.0 * np.power(x, -2.0)

# ==============================================================================
# 1. 工具函数 (无变化)
# ==============================================================================
def float_to_bf16_bits(f_val):
    return torch.tensor(f_val, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).item()

def float_to_fp32_bits(f_val):
    return np.array(f_val, dtype=np.float32).view(np.uint32).item()

def format_to_c_array(name, data, dtype="uint16_t"):
    if not isinstance(data, (list, np.ndarray)): return ""
    if dtype == "uint16_t":
        hex_data = [f"0x{float_to_bf16_bits(f):04x}" for f in data]
        c_type = "const uint16_t"
    elif dtype == "uint32_t":
        hex_data = [f"0x{float_to_fp32_bits(f):08x}" for f in data]
        c_type = "const uint32_t"
    else: raise ValueError("Unsupported dtype")
    s = f"{c_type} {name}[{len(hex_data)}] = {{\n    "
    for i, h in enumerate(hex_data):
        s += f"{h}, "
        if (i + 1) % 8 == 0 and i < len(hex_data) - 1: s += "\n    "
    s = s.strip().strip(",") + "\n}};\n"
    return s

# ==============================================================================
# 2. 系数生成器 (关键改进)
# ==============================================================================
def generate_pwl_uniform_optimized(func, range_min, range_max, num_segments, name):
    print(f"\n[GENERATE-PWL-OPTIMIZED] Generating coefficients for '{name}' using optimizer...")
    segment_width = (range_max - range_min) / num_segments
    k_coeffs, b_coeffs = [], []
    for i in range(num_segments):
        x_start = range_min + i * segment_width
        x_end = x_start + segment_width
        def error_func(p): return np.max(np.abs(p[0] * np.linspace(x_start, x_end, 50) + p[1] - func(np.linspace(x_start, x_end, 50))))
        k_init = (func(x_end) - func(x_start)) / (segment_width + 1e-12)
        b_init = func(x_start) - k_init * x_start
        res = minimize(error_func, [k_init, b_init], method='Nelder-Mead', tol=1e-9)
        k_coeffs.append(res.x[0]); b_coeffs.append(res.x[1])
    return np.array(k_coeffs), np.array(b_coeffs)

# [保留] 旧的泰勒展开方法，不再用于 rsqrt/reciprocal
def generate_pwl_exponent_based_taylor(func, func_deriv, num_segments, hw_exp_field_start, name):
    print(f"\n[GENERATE-PWL-HW-TAYLOR] Generating coefficients for '{name}' using robust Taylor method...")
    k_coeffs, b_coeffs = [], []
    for i in range(num_segments):
        exp_field = hw_exp_field_start + i
        real_exp = exp_field - 127
        x_start = 2.0**real_exp
        x_end = 2.0**(real_exp + 1.0)
        x_start_64, x_end_64 = np.float64(x_start), np.float64(x_end)
        x_mid_64 = np.sqrt(x_start_64 * x_end_64)
        y_mid_64 = func(x_mid_64)
        k = func_deriv(x_mid_64)
        b = y_mid_64 - k * x_mid_64
        k_coeffs.append(k)
        b_coeffs.append(b)
    return np.array(k_coeffs, dtype=np.float32), np.array(b_coeffs, dtype=np.float32)

# [关键改进] 新的、更高精度的方法，用于 rsqrt 和 reciprocal
def generate_pwl_exponent_based_optimized(func, num_segments, hw_exp_field_start, name):
    """
    结合了指数分段和优化器拟合的最高精度方法。
    """
    print(f"\n[GENERATE-PWL-HW-OPTIMIZED] Generating high-precision coefficients for '{name}' using segment-wise optimization...")
    k_coeffs, b_coeffs = [], []
    
    for i in range(num_segments):
        exp_field = hw_exp_field_start + i
        real_exp = exp_field - 127
        
        # 1. 使用指数方式定义每一个非均匀的段
        x_start = 2.0**real_exp
        x_end = 2.0**(real_exp + 1.0)

        # 2. 在这个段 [x_start, x_end] 内，使用优化器寻找最佳拟合直线
        #    这里的 error_func 是关键，它评估的是当前直线(由p=[k,b]定义)与真实函数在该段内的最大误差
        def error_func(p):
            # 在该段内取样50个点来评估误差
            x_samples = np.linspace(x_start, x_end, 50, dtype=np.float64)
            y_approx = p[0] * x_samples + p[1]
            y_true = func(x_samples)
            return np.max(np.abs(y_approx - y_true))

        # 3. 为优化器提供一个良好的初始猜测值 (使用过两端点的割线)
        k_init = (func(x_end) - func(x_start)) / (x_end - x_start + 1e-12)
        b_init = func(x_start) - k_init * x_start

        # 4. 运行优化器
        res = minimize(error_func, [k_init, b_init], method='Nelder-Mead', tol=1e-12)
        
        k_coeffs.append(res.x[0])
        b_coeffs.append(res.x[1])
        
    return np.array(k_coeffs, dtype=np.float32), np.array(b_coeffs, dtype=np.float32)

if __name__ == "__main__":
    
    # --- 为旧的 nlu_lookup 生成 bfloat16 系数 (用于 GELU 的 erf) ---
    print("="*60 + "\n      Generating bfloat16 LUTs for ERF\n" + "="*60)
    # 保留 bfloat16 的 exp 表生成，以防其他地方仍有旧调用，但这不再是 SiLU 的路径
    exp_k_bf16, exp_b_bf16 = generate_pwl_uniform_optimized(exp_func, EXP_RANGE[0], EXP_RANGE[1], NUM_SEGMENTS, "exp_func_bf16")
    erf_k_bf16, erf_b_bf16 = generate_pwl_uniform_optimized(erf_func, ERF_RANGE[0], ERF_RANGE[1], NUM_SEGMENTS, "erf_func_bf16")

    # --- [关键修改] 为所有高精度计算生成 float32 系数 ---
    print("\n" + "="*60 + "\n      Generating float32 LUTs for EXP, RSQRT, & RECIPROCAL\n" + "="*60)
    
    # [新增] 为 SiLU 的核心计算生成高精度的 float32 EXP 表
    exp_k_fp32, exp_b_fp32 = generate_pwl_uniform_optimized(exp_func, EXP_RANGE[0], EXP_RANGE[1], NUM_SEGMENTS, "exp_func_fp32")
    
    # [保留] 为 Norms 和 Softmax 生成高精度的 float32 RSQRT 和 RECIPROCAL 表
    rsqrt_k_fp32, rsqrt_b_fp32 = generate_pwl_exponent_based_optimized(rsqrt_func, NUM_SEGMENTS, HW_EXP_FIELD_START_RSQRT, "rsqrt_func")
    reciprocal_k_fp32, reciprocal_b_fp32 = generate_pwl_exponent_based_optimized(reciprocal_func, NUM_SEGMENTS, HW_EXP_FIELD_START_RECIPROCAL, "reciprocal_func")
    
    # --- 生成最终的、包含混合精度系数的C++头文件 ---
    print("\n[WRITE] Generating 'lut_tables.h' with final mixed-precision coefficients...")
    with open("lut_tables.h", "w") as f:
        f.write("// THIS FILE IS AUTO-GENERATED BY generate_luts.py (FINAL, MIXED-PRECISION)\n\n")
        f.write("#ifndef LUT_TABLES_H\n#define LUT_TABLES_H\n\n#include \"accelerator.h\"\n#include <cstdint>\n\n")
        f.write(f"#define NUM_SEGMENTS {NUM_SEGMENTS}\n\n")
        f.write(f"#define EXP_RANGE_MIN ({EXP_RANGE[0]}f)\n#define EXP_RANGE_MAX ({EXP_RANGE[1]}f)\n")
        f.write(f"#define EXP_SCALE_FACTOR ({(NUM_SEGMENTS / (EXP_RANGE[1] - EXP_RANGE[0]))}f)\n\n")
        f.write(f"#define ERF_RANGE_MIN ({ERF_RANGE[0]}f)\n#define ERF_RANGE_MAX ({ERF_RANGE[1]}f)\n")
        f.write(f"#define ERF_SCALE_FACTOR ({(NUM_SEGMENTS / (ERF_RANGE[1] - ERF_RANGE[0]))}f)\n\n")
        f.write(f"#define HW_EXP_FIELD_START_RSQRT {HW_EXP_FIELD_START_RSQRT}\n")
        f.write(f"#define HW_EXP_FIELD_START_RECIPROCAL {HW_EXP_FIELD_START_RECIPROCAL}\n\n")
        f.write(f"#define RSQRT_MIN_EXP_FIELD {HW_EXP_FIELD_START_RSQRT}\n")
        f.write(f"#define RECIPROCAL_MIN_EXP_FIELD {HW_EXP_FIELD_START_RECIPROCAL}\n\n")
        
        # --- bfloat16 系数表 (仅 GELU/ERF 使用) ---
        f.write("// --- BFLOAT16 Coefficients (for original nlu_lookup) ---\n")
        f.write(format_to_c_array("EXP_K", exp_k_bf16, dtype="uint16_t"))
        f.write(format_to_c_array("EXP_B", exp_b_bf16, dtype="uint16_t"))
        f.write(format_to_c_array("ERF_K", erf_k_bf16, dtype="uint16_t"))
        f.write(format_to_c_array("ERF_B", erf_b_bf16, dtype="uint16_t"))
        
        # --- float32 系数表 (用于所有高精度路径) ---
        f.write("\n// --- FLOAT32 Coefficients (for high-precision lookups) ---\n")
        
        # [新增] 写入新的 EXP_FP32 表
        f.write(format_to_c_array("EXP_K_FP32", exp_k_fp32, dtype="uint32_t"))
        f.write(format_to_c_array("EXP_B_FP32", exp_b_fp32, dtype="uint32_t"))

        # [保留] 写入 RSQRT 和 RECIPROCAL 表
        f.write(format_to_c_array("RSQRT_K_FP32", rsqrt_k_fp32, dtype="uint32_t"))
        f.write(format_to_c_array("RSQRT_B_FP32", rsqrt_b_fp32, dtype="uint32_t"))
        f.write(format_to_c_array("RECIPROCAL_K_FP32", reciprocal_k_fp32, dtype="uint32_t"))
        f.write(format_to_c_array("RECIPROCAL_B_FP32", reciprocal_b_fp32, dtype="uint32_t"))
        
        f.write("\n#endif // LUT_TABLES_H\n")
        
    print("\n[OK] Script finished successfully.")
    print("     'lut_tables.h' has been generated with mixed-precision tables.")

    