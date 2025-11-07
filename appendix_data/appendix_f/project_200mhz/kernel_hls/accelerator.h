#ifndef ACCELERATOR_TOP_H
#define ACCELERATOR_TOP_H

#include "hls_stream.h"
#include "ap_int.h"
#include <cstdint>
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef int32_t int32;
typedef int16_t int16;

// =================================================================
// 1. 核心数据类型与常量
// =================================================================
const int N_DIM = 64;
const int D_DIM = 768;

// 统一使用 ap_uint<16> 作为我们的 bfloat16 数据类型
typedef ap_uint<16> data_t;

// 激活函数操作码 (与你的原始代码保持一致)
enum ActivationOpcode {
    OP_ADD       = 0,
    OP_SOFTMAX   = 9,
    OP_SILU      = 4,
    OP_RMSNORM   = 5,
    OP_LAYERNORM = 6,
    OP_MUL       = 7,
    OP_GELU      = 8
};

// NLU 操作码 (供内部使用)
enum NluOp {
    NLU_OP_NO_OP,
    NLU_OP_RECIPROCAL,
    NLU_OP_RSQRT,
    NLU_OP_EXP,
    NLU_OP_ERF
};

const int BUS_WIDTH = 128;

// 根据总线位宽和数据类型计算并行因子
const int INTERFACE_FACTOR  = BUS_WIDTH / 16;       // = 8 for physical port
const int PARALLEL_FACTOR = INTERFACE_FACTOR;       // = 8 for inner computing virtual port

// AXI-MM port with 128 bit width
struct InterfaceBus {
    data_t data[INTERFACE_FACTOR];
};

// 定义宽总线数据类型，用于在顶层接口和内部数据流中传输 (wrong, the truth is virtual bus width)
struct WideBus {
    data_t data[PARALLEL_FACTOR];
};

void activation_accelerator(
    InterfaceBus* in0, // Renamed from in_a
    InterfaceBus* in1, // Renamed from in_b
    InterfaceBus* out,
    int stage,
    int config
);

#endif // ACCELERATOR_TOP_H