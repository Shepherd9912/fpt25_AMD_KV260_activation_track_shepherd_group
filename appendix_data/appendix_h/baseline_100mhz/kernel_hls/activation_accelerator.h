#ifndef ACTIVATION_ACCELERATOR_H
#define ACTIVATION_ACCELERATOR_H

#include <cstdint>

// Data type definitions for C simulation
typedef uint16_t uint16;
typedef int32_t int32;
typedef int64_t int64;

// bf16位运算加法函数声明
uint16 bf16add(uint16 a_bits, uint16 b_bits);

// Function declaration - 使用uint16数据类型
void activation_accelerator(uint16* in0, uint16* in1, uint16* out, int32 stage, int32 config);

// ===== 修改: 更新 DATA_SIZE =====
#define DATA_SIZE (64*768)

// Configuration definitions
#define CONFIG_ELTWISE_ADD 0     // Element-wise addition
#define CONFIG_SAFE_SOFTMAX 1    // Safe softmax activation function
#define CONFIG_MASK_SOFTMAX 2    // Masked softmax activation function
#define CONFIG_SIGMOID 3         // Sigmoid activation function
#define CONFIG_SILU 4            // SiLU (Swish) activation function
#define CONFIG_RMS_NORM 5        // RMS normalization
#define CONFIG_LAYER_NORM 6      // Layer normalization
// ===== 新增: 添加元素乘法定义 =====
#define CONFIG_ELTWISE_MUL 7     // Element-wise multiplication

// Stage definitions
#define STAGE_LOAD 0      // Data loading stage
#define STAGE_COMPUTE 1   // Computation stage
#define STAGE_STORE 2     // Data storage stage

#endif // ACTIVATION_ACCELERATOR_H