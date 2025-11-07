#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified script to generate all necessary data for the FPT'25 Activation Function Challenge.

This script generates:
1.  The primary 64x768 bf16 test tensor (X_test_tensor_bf16.bin).
2.  The corresponding 64x768 bf16 operand tensor for Add/Mul (Y_test_tensor_bf16.bin).
3.  A manifest.json file describing the contents of each row of the test tensor.
4.  Golden reference outputs for all 7 activation functions, computed directly from the
    generated test tensors, stored in a 'refs' subdirectory.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------
# SECTION 1: UTILITY FUNCTIONS
# ------------------------------

def nanmax(x: torch.Tensor, dim=None, keepdim=False):
    """NaN-aware max operation for torch tensors."""
    if dim is None:
        if torch.isnan(x).all(): return torch.tensor(float('nan'), dtype=x.dtype, device=x.device)
        xc = x.clone(); xc[torch.isnan(xc)] = -float('inf')
        return xc.max()
    else:
        xc = x.clone(); mask_nan = torch.isnan(xc); xc[mask_nan] = -float('inf')
        vals, _ = torch.max(xc, dim=dim, keepdim=keepdim)
        all_nan = mask_nan.all(dim=dim, keepdim=keepdim)
        return vals.masked_fill(all_nan, float('nan'))

def nansum(x: torch.Tensor, dim=None, keepdim=False):
    """NaN-aware sum operation."""
    xc = x.clone(); xc[torch.isnan(xc)] = 0.0
    return torch.sum(xc, dim=dim, keepdim=keepdim)

def nanmean(x: torch.Tensor, dim=None, keepdim=False):
    """NaN-aware mean operation."""
    mask = ~torch.isnan(x); xc = x.clone(); xc[~mask] = 0.0
    s = torch.sum(xc, dim=dim, keepdim=keepdim)
    cnt = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1)
    m = s / cnt
    if dim is None:
        return m if mask.any() else torch.tensor(float('nan'), dtype=x.dtype, device=x.device)
    else:
        return m.masked_fill(cnt == 0, float('nan'))

def to_bf16_bits(t_bf16: torch.Tensor) -> np.ndarray:
    """Converts a torch.bfloat16 tensor to a numpy.uint16 array of its bit patterns."""
    assert t_bf16.dtype == torch.bfloat16
    return t_bf16.view(torch.uint16).cpu().contiguous().numpy()

# ------------------------------
# SECTION 2: TEST VECTOR GENERATION LOGIC
# ------------------------------

# bf16 numerical constants
BF16_INFO = torch.finfo(torch.bfloat16)
BF16_MAX = float(BF16_INFO.max)
BF16_MIN_NORMAL = float(BF16_INFO.tiny)
SUBNORMAL_EXAMPLE_POS = 1e-39
SUBNORMAL_EXAMPLE_NEG = -1e-39

def build_manifest():
    # This function is copied from your script and defines the test case for each row.
    # (Content is collapsed for brevity, but it's the same as in your provided script)
    man = {
        0: "all +0.0", 1: "all +1.0", 2: "all -1.0", 3: "all +BF16_MAX", 4: "all -BF16_MAX",
        5: "all +BF16_MIN_NORMAL", 6: "all -BF16_MIN_NORMAL", 7: "all +subnormal (via 1e-40 -> bf16)",
        8: "all -subnormal (via -1e-40 -> bf16)", 9: "mixed normals pattern", 10: "mixed specials pattern",
        11: "alternating +0.0 / -0.0", 12: "uniform[-10,10]", 13: "uniform[-1000,1000]",
        14: "uniform[-0.1,0.1]", 15: "uniform[0,1]", 16: "normal(mean=0, std=5)",
        17: "uniform[-5,5] with injected +Inf / NaN", 18: "uniform[-5,5] with injected subnormals",
        19: "uniform[0, BF16_MAX/2]", 20: "softmax: large positives increasing",
        21: "softmax: large negatives decreasing", 22: "softmax: one large others 0.0",
        23: "softmax: one +Inf others 0.0", 24: "softmax: two +Inf others 0.0", 25: "softmax: one NaN others 0.0",
        26: "layernorm: [1,-1,...]", 27: "layernorm: near-constant", 28: "layernorm: random with +Inf / NaN",
        29: "layernorm: 0, eps, 2*eps, ...", 30: "layernorm: large values", 31: "layernorm: small normals",
        32: "rmsnorm: [1,-1,...]", 33: "rmsnorm: near zero", 34: "rmsnorm: random with +Inf / NaN",
        35: "rmsnorm: 0, eps, 2*eps, ...", 36: "rmsnorm: large values", 37: "rmsnorm: small normals",
        38: "SiLU: dense around zero", 39: "SiLU: large positives", 40: "SiLU: large negatives",
        41: "SiLU: random with +Inf/-Inf/NaN", 42: "GELU: dense around zero", 43: "GELU: large positives",
        44: "GELU: large negatives", 45: "GELU: random with +Inf/-Inf/NaN",
        46: "Ewise X: uniform[0,10]", 47: "Ewise X: uniform[-10,0]", 48: "Ewise X: large values for overflow",
        49: "Ewise X: small normals for underflow", 50: "Ewise X: random with +Inf",
        51: "Ewise X: random with -Inf", 52: "Ewise X: random with NaN", 53: "Ewise X: random with 0.0"
    }
    for i in range(54, 64): man[i] = "extra/mix patterns"
    return man

def fill_row(idx: int, D: int, rng: np.random.Generator) -> torch.Tensor:
    # This function is copied from your script and generates a single row of test data.
    # (Content is collapsed for brevity, but it's the same as in your provided script)
    # 用于在大范围内均匀取样
    lin_pos = torch.linspace(10.0, 100.0, D, dtype=torch.float32)
    lin_neg = torch.linspace(-10.0, -100.0, D, dtype=torch.float32)
    # 用于在 0点附近密集取样
    lin_zero = torch.linspace(-5.0, 5.0, D, dtype=torch.float32)

    # eps（LayerNorm 与 RMSNorm 的数值保护项用例）
    EPS_LN = 1e-4
    EPS_RMS = 1e-4
    def unif(a, b, s): return torch.from_numpy(rng.uniform(a, b, s).astype(np.float32))
    def normal(m, s, sz): return torch.from_numpy(rng.normal(m, s, sz).astype(np.float32))
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
# SECTION 3: GOLDEN REFERENCE IMPLEMENTATIONS (PyTorch)
# ------------------------------

def ref_softmax(x_bf16: torch.Tensor, dim=-1) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    x = x - nanmax(x, dim=dim, keepdim=True)
    y = torch.exp(x)
    return (y / nansum(y, dim=dim, keepdim=True)).to(torch.bfloat16)

def ref_layernorm(x_bf16: torch.Tensor, eps=1e-5) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    mean = nanmean(x, dim=-1, keepdim=True)
    var = nanmean((x - mean) ** 2, dim=-1, keepdim=True)
    return ((x - mean) / torch.sqrt(var + eps)).to(torch.bfloat16)

def ref_rmsnorm(x_bf16: torch.Tensor, eps=1e-5) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    msq = nanmean(x**2, dim=-1, keepdim=True)
    return (x / torch.sqrt(msq + eps)).to(torch.bfloat16)

def ref_silu(x_bf16: torch.Tensor) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    return (x * torch.sigmoid(x)).to(torch.bfloat16)

def ref_gelu(x_bf16: torch.Tensor) -> torch.Tensor:
    x = x_bf16.to(torch.float32)
    return F.gelu(x, approximate='tanh').to(torch.bfloat16)

def ref_add(x_bf16: torch.Tensor, y_bf16: torch.Tensor) -> torch.Tensor:
    return (x_bf16.to(torch.float32) + y_bf16.to(torch.float32)).to(torch.bfloat16)

def ref_mul(x_bf16: torch.Tensor, y_bf16: torch.Tensor) -> torch.Tensor:
    return (x_bf16.to(torch.float32) * y_bf16.to(torch.float32)).to(torch.bfloat16)


# ------------------------------
# SECTION 4: MAIN EXECUTION
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate all test vectors and golden references for FPT'25.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for all generated files.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument("--N", type=int, default=64, help="Number of rows in the test tensor.")
    parser.add_argument("--D", type=int, default=768, help="Number of columns in the test tensor.")
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.outdir, exist_ok=True)
    ref_dir = os.path.join(args.outdir, "refs")
    os.makedirs(ref_dir, exist_ok=True)
    
    print(f"Starting data generation in directory: {args.outdir}")
    print(f"Tensor dimensions: {args.N}x{args.D}, Seed: {args.seed}")

    # --- 1. Generate primary test vectors X and Y ---
    rng = np.random.default_rng(args.seed)
    # IMPORTANT: The following line is a placeholder. You MUST replace the body of the `fill_row` and `build_Y_for_ewise`
    # functions above with your complete, detailed logic from your original script.
    X_f32 = torch.stack([fill_row(i, args.D, rng) for i in range(args.N)], dim=0)
    X_bf16 = X_f32.to(torch.bfloat16)
    Y_bf16 = build_Y_for_ewise(X_bf16, rng)
    print("Step 1: Primary test tensors X and Y generated.")

    # --- 2. Save primary test vectors ---
    to_bf16_bits(X_bf16).tofile(os.path.join(args.outdir, "X_test_tensor_bf16.bin"))
    to_bf16_bits(Y_bf16).tofile(os.path.join(args.outdir, "Y_test_tensor_bf16.bin"))
    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(build_manifest(), f, ensure_ascii=False, indent=2)
    print("Step 2: Saved X_test_tensor_bf16.bin, Y_test_tensor_bf16.bin, and manifest.json.")

    # --- 3. Compute and save all golden references ---
    print("Step 3: Generating and saving golden reference files...")
    ref_functions = {
        "softmax": (ref_softmax, (X_bf16,)),
        "layernorm": (ref_layernorm, (X_bf16,)),
        "rmsnorm": (ref_rmsnorm, (X_bf16,)),
        "silu": (ref_silu, (X_bf16,)),
        "gelu": (ref_gelu, (X_bf16,)),
        "add": (ref_add, (X_bf16, Y_bf16)),
        "mul": (ref_mul, (X_bf16, Y_bf16)),
    }

    for name, (func, args_tuple) in ref_functions.items():
        ref_bf16 = func(*args_tuple)
        
        # Save .pt file
        pt_path = os.path.join(ref_dir, f"ref_{name}_bf16.pt")
        torch.save(ref_bf16, pt_path)
        
        # Save .bin file
        bin_path = os.path.join(ref_dir, f"ref_{name}_bf16.bin")
        to_bf16_bits(ref_bf16).tofile(bin_path)
        print(f"  - Generated and saved references for {name}.")

    print("\n--- Generation Complete! ---")
    print("Summary of created files:")
    print(f"- {args.outdir}/X_test_tensor_bf16.bin (Input)")
    print(f"- {args.outdir}/Y_test_tensor_bf16.bin (Input for Add/Mul)")
    print(f"- {args.outdir}/manifest.json (Test case descriptions)")
    print(f"- {ref_dir}/ (Contains all 7 golden reference files in .pt and .bin format)")


if __name__ == "__main__":
    main()
"""
python3 gen_test_and_golden_data.py --outdir ../../on_board
"""