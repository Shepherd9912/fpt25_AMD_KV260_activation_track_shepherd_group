#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 bf16 测试向量（64x768）与配套的 Y（Add/Mul），以及可选的 PyTorch 参考输出。
- X_test_tensor: torch.bfloat16, shape = (64, 768)
- Y_test_tensor: torch.bfloat16, shape = (64, 768)
- 导出 bf16 位模式（uint16）为 .bin 文件，方便 C++ Testbench 加载
- 可选 (--emit_ref)：为七类算子生成参考输出 .bin 文件
"""

import os
import json
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------
# 工具函数：bf16 <-> 位模式 (这部分无需修改)
# ------------------------------
def nanmax(x: torch.Tensor, dim=None, keepdim=False):
    if dim is None:
        if torch.isnan(x).all(): return torch.tensor(float('nan'), dtype=x.dtype, device=x.device)
        xc = x.clone(); xc[torch.isnan(xc)] = -float('inf'); return xc.max()
    else:
        xc = x.clone(); mask_nan = torch.isnan(xc); xc[mask_nan] = -float('inf')
        vals, idx = torch.max(xc, dim=dim, keepdim=keepdim)
        all_nan = mask_nan.all(dim=dim, keepdim=keepdim)
        vals = vals.masked_fill(all_nan, float('nan')); return vals

def nansum(x: torch.Tensor, dim=None, keepdim=False):
    xc = x.clone(); xc[torch.isnan(xc)] = 0.0; return torch.sum(xc, dim=dim, keepdim=keepdim)

def nanmean(x: torch.Tensor, dim=None, keepdim=False):
    mask = ~torch.isnan(x); xc = x.clone(); xc[~mask] = 0.0
    s = torch.sum(xc, dim=dim, keepdim=keepdim); cnt = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)
    m = s / cnt
    if dim is None: return m if mask.any() else torch.tensor(float('nan'), dtype=x.dtype, device=x.device)
    else: return m.masked_fill(cnt == 0, float('nan'))

def to_bf16_bits(t_bf16: torch.Tensor) -> np.ndarray:
    """
    将 torch.bfloat16 Tensor 的位模式导出为 numpy.uint16（行主序）。
    """
    assert t_bf16.dtype == torch.bfloat16
    # 使用 torch.short 替代 torch.uint16 以兼容旧版本PyTorch
    # torch.short 对应于有符号16位整数，但在这里我们只关心它的位模式，
    # 最终通过 .numpy() 转换到 numpy.uint16 时，位模式会保持不变。
    return t_bf16.view(torch.short).cpu().contiguous().numpy().astype(np.uint16)

# ... (从 from_bf16_bits 到 build_Y_for_ewise 的所有函数都保持不变) ...
# (此处省略了未修改的函数体，以保持简洁)
def from_bf16_bits(bits_u16: np.ndarray, shape) -> torch.Tensor:
    t = torch.from_numpy(bits_u16.astype(np.uint16)).view(shape)
    return t.view(torch.bfloat16)

def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    return x.to(torch.bfloat16)

def bf16_to_f32(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    return x.to(torch.float32)

BF16_INFO = torch.finfo(torch.bfloat16)
BF16_MAX = float(BF16_INFO.max)
BF16_MIN_NORMAL = float(BF16_INFO.tiny)
SUBNORMAL_EXAMPLE_POS = 1e-39
SUBNORMAL_EXAMPLE_NEG = -1e-39

def build_manifest():
    # ... (函数体不变) ...
    man = {}
    # 0–11: 通用/特殊
    man.update({
        0:  "all +0.0",
        1:  "all +1.0",
        2:  "all -1.0",
        3:  "all +BF16_MAX",
        4:  "all -BF16_MAX",
        5:  "all +BF16_MIN_NORMAL",
        6:  "all -BF16_MIN_NORMAL",
        7:  "all +subnormal (via 1e-40 -> bf16)",
        8:  "all -subnormal (via -1e-40 -> bf16)",
        9:  "mixed normals: [1,-1,0.5,-0.5,2,-2,0.1,-0.1,10,-10] pattern",
        10: "mixed specials: [+Inf,-Inf,NaN,+0.0,-0.0,1,-1] pattern",
        11: "alternating +0.0 / -0.0",
    })
    # 12–19: 随机分布
    man.update({
        12: "uniform[-10,10]",
        13: "uniform[-1000,1000]",
        14: "uniform[-0.1,0.1]",
        15: "uniform[0,1]",
        16: "normal(mean=0, std=5)",
        17: "uniform[-5,5] with injected +Inf / NaN",
        18: "uniform[-5,5] with injected subnormals",
        19: "uniform[0, BF16_MAX/2]",
    })
    # 20–25: Softmax 特定
    man.update({
        20: "softmax: large positives increasing (100 + 0.1*k)",
        21: "softmax: large negatives decreasing (-100 - 0.1*k)",
        22: "softmax: one large (10.0) others 0.0",
        23: "softmax: one +Inf others 0.0",
        24: "softmax: two +Inf others 0.0",
        25: "softmax: one NaN others 0.0",
    })
    # 26–31: LayerNorm
    man.update({
        26: "layernorm: [1,-1,1,-1,...]",
        27: "layernorm: near-constant (5.000, 5.001, ...)",
        28: "layernorm: random with +Inf / NaN injected",
        29: "layernorm: 0, eps, 2*eps, ...",
        30: "layernorm: large values uniform[0, BF16_MAX/10]",
        31: "layernorm: small normals uniform[MIN_N, 10*MIN_N]",
    })
    # 32–37: RMSNorm
    man.update({
        32: "rmsnorm: [1,-1,1,-1,...]",
        33: "rmsnorm: near zero (0.000, 0.001, ...)",
        34: "rmsnorm: random with +Inf / NaN injected",
        35: "rmsnorm: 0, eps, 2*eps, ...",
        36: "rmsnorm: large values uniform[0, BF16_MAX/10]",
        37: "rmsnorm: small normals uniform[MIN_N, 10*MIN_N]",
    })
    # 38–41: SiLU
    man.update({
        38: "SiLU: dense around zero (linspace -5..5)",
        39: "SiLU: large positives (linspace 10..100)",
        40: "SiLU: large negatives (linspace -10..-100)",
        41: "SiLU: random with +Inf/-Inf/NaN injected",
    })
    # 42–45: GELU
    man.update({
        42: "GELU: dense around zero (linspace -5..5)",
        43: "GELU: large positives (linspace 10..100)",
        44: "GELU: large negatives (linspace -10..-100)",
        45: "GELU: random with +Inf/-Inf/NaN injected",
    })
    # 46–53: Elementwise Add/Mul（X 行；Y 由脚本配对生成）
    man.update({
        46: "Ewise X: uniform[0,10]   (pair Y for general add/mul)",
        47: "Ewise X: uniform[-10,0]  (pair Y for general add/mul)",
        48: "Ewise X: large values uniform[BF16_MAX/2, BF16_MAX] (overflow tests)",
        49: "Ewise X: small normals uniform[MIN_N, 10*MIN_N]     (underflow tests)",
        50: "Ewise X: random with +Inf injected",
        51: "Ewise X: random with -Inf injected",
        52: "Ewise X: random with NaN injected",
        53: "Ewise X: random with 0.0 injected",
    })
    # 54–63: 其他扩展/病态
    for i in range(54, 64):
        man[i] = "extra/mix patterns (lognormal / alternating extremes / different seeds)"
    return man

def fill_row(idx: int, D: int, rng: np.random.Generator) -> torch.Tensor:
    # ... (函数体不变) ...
    # 常用构件
    lin01 = torch.linspace(0, 1, D, dtype=torch.float32)
    # 用于在大范围内均匀取样
    lin_pos = torch.linspace(10.0, 100.0, D, dtype=torch.float32)
    lin_neg = torch.linspace(-10.0, -100.0, D, dtype=torch.float32)
    # 用于在 0点附近密集取样
    lin_zero = torch.linspace(-5.0, 5.0, D, dtype=torch.float32)

    # eps（LayerNorm 与 RMSNorm 的数值保护项用例）
    EPS_LN = 1e-4
    EPS_RMS = 1e-4

    # 便捷随机函数（用 numpy 生成，再转 torch）
    def unif(a, b, size):
        return torch.from_numpy(rng.uniform(a, b, size).astype(np.float32))
    def normal(mean, std, size):
        return torch.from_numpy(rng.normal(mean, std, size).astype(np.float32))

    # ---- 分类生成 ----
    if idx == 0:
        return torch.zeros(D, dtype=torch.float32)
    if idx == 1:
        return torch.ones(D, dtype=torch.float32)
    if idx == 2:
        return -torch.ones(D, dtype=torch.float32)
    if idx == 3:
        return torch.full((D,), BF16_MAX, dtype=torch.float32)
    if idx == 4:
        return torch.full((D,), -BF16_MAX, dtype=torch.float32)
    if idx == 5:
        return torch.full((D,), BF16_MIN_NORMAL, dtype=torch.float32)
    if idx == 6:
        return torch.full((D,), -BF16_MIN_NORMAL, dtype=torch.float32)
    if idx == 7:
        return torch.full((D,), SUBNORMAL_EXAMPLE_POS, dtype=torch.float32)
    if idx == 8:
        return torch.full((D,), SUBNORMAL_EXAMPLE_NEG, dtype=torch.float32)
    if idx == 9:
        base = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1, 10.0, -10.0], dtype=torch.float32)
        return base.repeat(D // base.numel() + 1)[:D].clone()
    if idx == 10:
        base = torch.tensor([float('inf'), float('-inf'), float('nan'), 0.0, -0.0, 1.0, -1.0], dtype=torch.float32)
        return base.repeat(D // base.numel() + 1)[:D].clone()
    if idx == 11:
        a = torch.zeros(D, dtype=torch.float32) # 全部 +0.0
        a[1::2] = -0.0 # 奇数位置 -0.0
        return a

    if idx == 12:
        return unif(-10.0, 10.0, D)
    if idx == 13:
        return unif(-1000.0, 1000.0, D)
    if idx == 14:
        return unif(-0.1, 0.1, D)
    if idx == 15:
        return unif(0.0, 1.0, D)
    if idx == 16:
        return normal(0.0, 5.0, D)
    if idx == 17:
        a = unif(-5.0, 5.0, D)
        # 注入若干 Inf / NaN
        inj = min(8, D)       # 计划注入个数为8个
        pos = rng.choice(D, size=inj, replace=False)   # 不重复位置，随机选择位置注入
        for k, p in enumerate(pos):
            a[p] = float('inf') if k % 2 == 0 else float('nan')
        return a
    if idx == 18:
        a = unif(-5.0, 5.0, D)
        inj = min(8, D)
        pos = rng.choice(D, size=inj, replace=False)
        for p in pos:
            a[p] = SUBNORMAL_EXAMPLE_POS if rng.random() < 0.5 else SUBNORMAL_EXAMPLE_NEG
        return a
    if idx == 19:
        return unif(0.0, BF16_MAX/2.0, D)

    if idx == 20:
        return 100.0 + 0.1 * torch.arange(D, dtype=torch.float32)
    if idx == 21:
        return -100.0 - 0.1 * torch.arange(D, dtype=torch.float32)
    if idx == 22:
        a = torch.zeros(D, dtype=torch.float32)
        a[D//2] = 10.0 # 中间索引为10
        return a
    if idx == 23:
        a = torch.zeros(D, dtype=torch.float32)
        a[D//2] = float('inf')
        return a
    if idx == 24:
        a = torch.zeros(D, dtype=torch.float32)
        a[D//3] = float('inf'); a[2*D//3] = float('inf')
        return a
    if idx == 25:
        a = torch.zeros(D, dtype=torch.float32)
        a[D//2] = float('nan')
        return a

    if idx == 26:
        base = torch.tensor([1.0, -1.0], dtype=torch.float32)
        return base.repeat(D // 2 + 1)[:D].clone()
    if idx == 27:
        # 5.000, 5.001, 5.002, ...
        return 5.0 + 0.001 * torch.arange(D, dtype=torch.float32)
    if idx == 28:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        for k, p in enumerate(pos):
            a[p] = float('inf') if k % 3 == 0 else float('nan')
        return a
    if idx == 29:
        return torch.arange(D, dtype=torch.float32) * EPS_LN
    if idx == 30:
        return unif(0.0, BF16_MAX/10.0, D)
    if idx == 31:
        return unif(BF16_MIN_NORMAL, BF16_MIN_NORMAL*10.0, D)

    if idx == 32:
        base = torch.tensor([1.0, -1.0], dtype=torch.float32)
        return base.repeat(D // 2 + 1)[:D].clone()
    if idx == 33:
        return torch.arange(D, dtype=torch.float32) * 0.001
    if idx == 34:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        for k, p in enumerate(pos):
            a[p] = float('inf') if k % 3 == 0 else float('nan')
        return a
    if idx == 35:
        return torch.arange(D, dtype=torch.float32) * EPS_RMS
    if idx == 36:
        return unif(0.0, BF16_MAX/10.0, D)
    if idx == 37:
        return unif(BF16_MIN_NORMAL, BF16_MIN_NORMAL*10.0, D)

    if idx == 38:
        return lin_zero
    if idx == 39:
        return lin_pos
    if idx == 40:
        return lin_neg
    if idx == 41:
        a = unif(-5.0, 5.0, D)
        # 注入 +Inf / -Inf / NaN
        pos = rng.choice(D, size=min(12, D), replace=False)
        for i, p in enumerate(pos):
            a[p] = float('inf') if i % 3 == 0 else (float('-inf') if i % 3 == 1 else float('nan'))
        return a

    if idx == 42:
        return lin_zero
    if idx == 43:
        return lin_pos
    if idx == 44:
        return lin_neg
    if idx == 45:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(12, D), replace=False)
        for i, p in enumerate(pos):
            a[p] = float('inf') if i % 3 == 0 else (float('-inf') if i % 3 == 1 else float('nan'))
        return a

    # 46–53: 仅作为 Ewise X
    if idx == 46:
        return unif(0.0, 10.0, D)
    if idx == 47:
        return unif(-10.0, 0.0, D)
    if idx == 48:
        return unif(BF16_MAX/2.0, BF16_MAX, D)
    if idx == 49:
        return unif(BF16_MIN_NORMAL, BF16_MIN_NORMAL*10.0, D)
    if idx == 50:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        a[pos] = float('inf')
        return a
    if idx == 51:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        a[pos] = float('-inf')
        return a
    if idx == 52:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        a[pos] = float('nan')
        return a
    if idx == 53:
        a = unif(-5.0, 5.0, D)
        pos = rng.choice(D, size=min(8, D), replace=False)
        a[pos] = 0.0
        return a

    # 54–63：扩展/病态
    if idx in range(54, 64):
        choice = idx % 3
        if choice == 0:
            # 对数正态（偏态分布）
            a = np.random.lognormal(mean=0.0, sigma=1.0, size=D).astype(np.float32)
            return torch.from_numpy(a)
        elif choice == 1:
            # 交替极值
            a = torch.zeros(D, dtype=torch.float32)
            a[::2] = BF16_MAX
            a[1::2] = -BF16_MAX
            return a
        else:
            # 不同种子随机
            local_rng = np.random.default_rng(1000 + idx)
            return torch.from_numpy(local_rng.uniform(-50.0, 50.0, D).astype(np.float32))

    raise ValueError(f"Unhandled row index {idx}")

def build_Y_for_ewise(X: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    # ... (函数体不变) ...
    N, D = X.shape
    Y = torch.zeros_like(X, dtype=torch.float32)

    # 46: X in [0,10] → Y in [0,10]
    Y[46] = torch.from_numpy(rng.uniform(0.0, 10.0, D).astype(np.float32))

    # 47: X in [-10,0] → Y in [-10,0]
    Y[47] = torch.from_numpy(rng.uniform(-10.0, 0.0, D).astype(np.float32))

    # 48: X large (MAX/2 .. MAX) → Y large (MAX/2 .. MAX) 触发溢出
    Y[48] = torch.from_numpy(rng.uniform(BF16_MAX/2.0, BF16_MAX, D).astype(np.float32))

    # 49: X small normals → Y small normals 触发下溢/次正规
    Y[49] = torch.from_numpy(rng.uniform(BF16_MIN_NORMAL, BF16_MIN_NORMAL*10.0, D).astype(np.float32))

    # 50: X 含 +Inf → Y 含 -Inf，测试 Inf + (-Inf) 与 0*Inf
    Y[50] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[50, pos] = float('-inf')
    # 同时在少数位置放 0，便于 0*Inf 测试
    pos0 = rng.choice(D, size=min(8, D), replace=False)
    Y[50, pos0] = 0.0

    # 51: X 含 -Inf → Y 含 +Inf
    Y[51] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[51, pos] = float('inf')

    # 52: X 含 NaN → Y 任意；也额外注入少量 NaN，看传播
    Y[52] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[52, pos] = float('nan')

    # 53: X 含 0.0 → Y 含 Inf，测试 0*Inf = NaN
    Y[53] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[53, pos] = float('inf')

    return Y.to(torch.bfloat16)  # 统一 bf16


# ------------------------------
# 参考实现（PyTorch）
# ------------------------------
def ref_softmax(x_bf16: torch.Tensor, dim=-1) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    x = x - nanmax(x, dim=dim, keepdim=True)
    y = torch.exp(x)
    denom = nansum(y, dim=dim, keepdim=True)
    out = y / denom
    return out.to(torch.bfloat16)

def ref_layernorm(x_bf16: torch.Tensor, eps=1e-5) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    mean = nanmean(x, dim=-1, keepdim=True)
    var = nanmean((x - mean) ** 2, dim=-1, keepdim=True)
    out = (x - mean) / torch.sqrt(var + eps)
    return out.to(torch.bfloat16)

def ref_rmsnorm(x_bf16: torch.Tensor, eps=1e-5) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    msq = nanmean(x**2, dim=-1, keepdim=True)
    out = x / torch.sqrt(msq + eps)
    return out.to(torch.bfloat16)

def ref_silu(x_bf16: torch.Tensor) -> torch.Tensor:
    # 直接用 bf16 计算以匹配硬件行为
    return (x_bf16.float() * torch.sigmoid(x_bf16.float())).to(torch.bfloat16)

def ref_gelu(x_bf16: torch.Tensor) -> torch.Tensor:
    # GELU 的标准近似公式 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    # 为简单起见，我们使用 PyTorch 内置的，它有 tanh 近似
    return F.gelu(x_bf16.float(), approximate='tanh').to(torch.bfloat16)

def ref_add(x_bf16: torch.Tensor, y_bf16: torch.Tensor) -> torch.Tensor:
    # PyTorch 会自动提升到 float32 计算
    return (x_bf16 + y_bf16).to(torch.bfloat16)

def ref_mul(x_bf16: torch.Tensor, y_bf16: torch.Tensor) -> torch.Tensor:
    return (x_bf16 * y_bf16).to(torch.bfloat16)

# ==============================================================================
# ===== 新增：将 numpy 数组写入 C++ 头文件的辅助函数 =====
# ==============================================================================
def write_to_cpp_header(data_u16: np.ndarray, var_name: str, file_handle, items_per_line=16):
    """将 uint16 numpy 数组格式化为 C++ const 数组并写入文件"""
    file_handle.write(f"const uint16_t {var_name}[{len(data_u16)}] = {{\n    ")
    for i, val in enumerate(data_u16):
        file_handle.write(f"0x{val:04x}, ")
        if (i + 1) % items_per_line == 0 and (i + 1) < len(data_u16):
            file_handle.write("\n    ")
    file_handle.write("\n};\n\n")

# ==============================================================================
# ===== 主流程 (Main) - 替换为这个新版本 =====
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate a C++ header with bf16 test vectors and golden outputs.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for the generated header file.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument("--N", type=int, default=64, help="Number of rows.")
    parser.add_argument("--D", type=int, default=768, help="Dimension of each row.")
    args = parser.parse_args()

    output_header_path = os.path.join(args.outdir, "test_data.h")
    print(f"Generating C++ header file: {output_header_path}")

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    N, D = args.N, args.D

    # 1) 生成 X 和 Y
    print("Generating X and Y tensors...")
    X_f32 = torch.stack([fill_row(i, D, rng) for i in range(N)], dim=0)
    X_bf16 = f32_to_bf16(X_f32)
    Y_bf16 = build_Y_for_ewise(X_bf16, rng)
    
    X_bits = to_bf16_bits(X_bf16).flatten()
    Y_bits = to_bf16_bits(Y_bf16).flatten()

    # 2) 计算所有黄金参考
    print("Calculating all 7 golden references...")
    golden_tasks = {
        "add":       (ref_add,       (X_bf16, Y_bf16)),
        "mul":       (ref_mul,       (X_bf16, Y_bf16)),
        "gelu":      (ref_gelu,      (X_bf16,)),
        "silu":      (ref_silu,      (X_bf16,)),
        "softmax":   (ref_softmax,   (X_bf16,)),
        "layernorm": (ref_layernorm, (X_bf16,)),
        "rmsnorm":   (ref_rmsnorm,   (X_bf16,))
    }
    
    golden_results_bits = {}
    for name, (ref_func, args_tuple) in golden_tasks.items():
        golden_output_bf16 = ref_func(*args_tuple)
        golden_results_bits[name] = to_bf16_bits(golden_output_bf16).flatten()

    # 3) 写入头文件
    with open(output_header_path, "w") as f:
        f.write("#ifndef TEST_DATA_H\n")
        f.write("#define TEST_DATA_H\n\n")
        f.write("#include <cstdint>\n\n")
        
        print("  - Writing input X...")
        write_to_cpp_header(X_bits, "hls_input_X", f)
        
        print("  - Writing input Y...")
        write_to_cpp_header(Y_bits, "hls_input_Y", f)
        
        for name, data in golden_results_bits.items():
            print(f"  - Writing golden_{name}...")
            write_to_cpp_header(data, f"hls_golden_{name}", f)
            
        f.write("#endif // TEST_DATA_H\n")

    print(f"\n[OK] Successfully created '{output_header_path}'")
    print("You can now #include this file in your HLS testbench.")

if __name__ == "__main__":
    main()
"""
python3 generate_testbench_data.py --outdir ../
"""