#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成 bf16 测试向量（64x768）与配套的 Y，以及 PyTorch 参考输出。
- X_test_tensor (in0): torch.bfloat16, shape = (64, 768)
- Y_test_tensor (in1): torch.bfloat16, shape = (64, 768)
- mask_tensor (mask): torch.bfloat16, shape = (64, 768)
- 导出 bf16 位模式（uint16）为 .bin 文件，方便 C++ Testbench 加载。
- 为 HLS 代码中定义的七类算子生成参考输出 .bin 文件。
"""

import os
import argparse
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
    return t_bf16.view(torch.short).cpu().contiguous().numpy().astype(np.uint16)

def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    return x.to(torch.bfloat16)

BF16_INFO = torch.finfo(torch.bfloat16)
BF16_MAX = float(BF16_INFO.max)
BF16_MIN_NORMAL = float(BF16_INFO.tiny)
SUBNORMAL_EXAMPLE_POS = 1e-39
SUBNORMAL_EXAMPLE_NEG = -1e-39

def fill_row(idx: int, D: int, rng: np.random.Generator) -> torch.Tensor:
    # (此函数内容与您提供的版本完全相同，为简洁此处省略)
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

    if idx == 42: # GELU (can reuse SiLU patterns)
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
            a = np.random.lognormal(mean=0.0, sigma=1.0, size=D).astype(np.float32)
            return torch.from_numpy(a)
        elif choice == 1:
            a = torch.zeros(D, dtype=torch.float32)
            a[::2] = BF16_MAX
            a[1::2] = -BF16_MAX
            return a
        else:
            local_rng = np.random.default_rng(1000 + idx)
            return torch.from_numpy(local_rng.uniform(-50.0, 50.0, D).astype(np.float32))

    raise ValueError(f"Unhandled row index {idx}")

def build_Y_for_ewise(X: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    # (此函数内容与您提供的版本完全相同，为简洁此处省略)
    N, D = X.shape
    Y = torch.zeros_like(X, dtype=torch.float32)
    Y[46] = torch.from_numpy(rng.uniform(0.0, 10.0, D).astype(np.float32))
    Y[47] = torch.from_numpy(rng.uniform(-10.0, 0.0, D).astype(np.float32))
    Y[48] = torch.from_numpy(rng.uniform(BF16_MAX/2.0, BF16_MAX, D).astype(np.float32))
    Y[49] = torch.from_numpy(rng.uniform(BF16_MIN_NORMAL, BF16_MIN_NORMAL*10.0, D).astype(np.float32))
    Y[50] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[50, pos] = float('-inf')
    pos0 = rng.choice(D, size=min(8, D), replace=False)
    Y[50, pos0] = 0.0
    Y[51] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[51, pos] = float('inf')
    Y[52] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[52, pos] = float('nan')
    Y[53] = torch.from_numpy(rng.uniform(-5.0, 5.0, D).astype(np.float32))
    pos = rng.choice(D, size=min(8, D), replace=False)
    Y[53, pos] = float('inf')
    return Y.to(torch.bfloat16)

# ------------------------------
# 参考实现（PyTorch）
# ------------------------------
def ref_add(x_bf16: torch.Tensor, y_bf16: torch.Tensor) -> torch.Tensor:
    return (x_bf16.to(torch.float32) + y_bf16.to(torch.float32)).to(torch.bfloat16)

def ref_softmax(x_bf16: torch.Tensor, dim=-1) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    x = x - nanmax(x, dim=dim, keepdim=True)
    y = torch.exp(x)
    denom = nansum(y, dim=dim, keepdim=True)
    out = y / denom
    return out.to(torch.bfloat16)

def ref_sigmoid(x_bf16: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x_bf16.to(torch.float32)).to(torch.bfloat16)

def ref_silu(x_bf16: torch.Tensor) -> torch.Tensor:
    return F.silu(x_bf16.to(torch.float32)).to(torch.bfloat16)

def ref_rmsnorm(x_bf16: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    msq = nanmean(x**2, dim=-1, keepdim=True)
    out = x / torch.sqrt(msq + eps)
    return out.to(torch.bfloat16)

def ref_layernorm(x_bf16: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    mean = nanmean(x, dim=-1, keepdim=True)
    var = nanmean((x - mean) ** 2, dim=-1, keepdim=True)
    out = (x - mean) / torch.sqrt(var + eps)
    return out.to(torch.bfloat16)

def ref_mul(x_bf16: torch.Tensor, y_bf16: torch.Tensor) -> torch.Tensor:
    return (x_bf16.to(torch.float32) * y_bf16.to(torch.float32)).to(torch.bfloat16)

# ==============================================================================
# ===== 新增：将 numpy 数组写入 .bin 文件的辅助函数 =====
# ==============================================================================
def save_to_bin_file(data_u16: np.ndarray, path: str):
    """将 uint16 numpy 数组的内容直接写入二进制文件。"""
    data_u16.tofile(path)

# ==============================================================================
# ===== 主流程 (Main) - 替换为这个新版本 =====
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate bf16 .bin test vectors and golden outputs for HLS testbench.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for the generated .bin files.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument("--N", type=int, default=64, help="Number of rows (Batch size).")
    parser.add_argument("--D", type=int, default=768, help="Dimension of each row (Tensor size).")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Generating binary files in: {args.outdir}")

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    N, D = args.N, args.D

    # 1) 生成 X, Y 和 Mask
    print("Generating X (in0), Y (in1), and Mask tensors...")
    X_f32 = torch.stack([fill_row(i, D, rng) for i in range(N)], dim=0)
    X_bf16 = f32_to_bf16(X_f32)
    Y_bf16 = build_Y_for_ewise(X_bf16, rng)
    # 生成一个全1的Mask作为占位符
    mask_bf16 = torch.ones_like(X_bf16)

    # 保存输入文件
    save_to_bin_file(to_bf16_bits(X_bf16), os.path.join(args.outdir, "in0_bf16.bin"))
    save_to_bin_file(to_bf16_bits(Y_bf16), os.path.join(args.outdir, "in1_bf16.bin"))
    save_to_bin_file(to_bf16_bits(mask_bf16), os.path.join(args.outdir, "mask_bf16.bin"))
    print("  - Saved in0_bf16.bin, in1_bf16.bin, mask_bf16.bin")

    # 2) 计算并保存所有黄金参考文件
    print("\nCalculating and saving all 7 golden reference files...")

    # 定义 HLS config 索引与参考实现的映射关系
    # 格式: (config_idx, ref_function, args_tuple)
    golden_configs = [
        (0, ref_add,       (X_bf16, Y_bf16)),
        (1, ref_softmax,   (X_bf16,)),
        # config 2 is mask_safe_softmax, which is disabled in HLS.
        # We generate a zero file as a placeholder for the testbench.
        (2, None,          None),
        (3, ref_sigmoid,   (X_bf16,)),
        (4, ref_silu,      (X_bf16,)),
        (5, ref_rmsnorm,   (X_bf16,)),
        (6, ref_layernorm, (X_bf16,)),
        (7, ref_mul,       (X_bf16, Y_bf16))
    ]

    for config_idx, ref_func, args_tuple in golden_configs:
        output_filename = f"golden_out_config_{config_idx}_bf16.bin"
        output_path = os.path.join(args.outdir, output_filename)
        
        if ref_func is None:
            # 特殊处理 config 2
            print(f"  - Generating placeholder (zeros) for config {config_idx} -> {output_filename}")
            golden_output_bf16 = torch.zeros_like(X_bf16)
        else:
            print(f"  - Calculating config {config_idx} ({ref_func.__name__}) -> {output_filename}")
            golden_output_bf16 = ref_func(*args_tuple)
        
        save_to_bin_file(to_bf16_bits(golden_output_bf16), output_path)

    print(f"\n[OK] Successfully created all test data in '{args.outdir}'")

if __name__ == "__main__":
    main()
