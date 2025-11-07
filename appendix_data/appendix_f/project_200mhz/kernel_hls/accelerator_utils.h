#ifndef ACCELERATOR_UTILS_H
#define ACCELERATOR_UTILS_H
// 适配v0_7之前的所有版本
#include "ap_int.h"
#include "hls_stream.h"
#include "hls_math.h"
#include "accelerator.h" // For data_t

// ===================================================================================
//  This file contains all low-level, shared utilities and constants.
// ===================================================================================

union FloatUIntUnion { float f; ap_uint<32> u; };
// --- 新增：使用 bit_cast 的类型转换函数 ---
template<typename T, typename U>
U bit_cast(T x) {
    return *reinterpret_cast<U*>(&x);
}

// --- Core Data Conversion ---
inline ap_uint<32> bf16_2_fp32(data_t x) { 
    #pragma HLS INLINE
    return (ap_uint<32>)x << 16; }
inline data_t fp32_2_bf16(ap_uint<32> x) {
    #pragma HLS INLINE
    ap_uint<16> lsbs = x;
    ap_uint<32> rounded_x = x + 0x00008000;
    return (data_t)(rounded_x >> 16);
}
inline data_t negate_bf16(data_t x) { 
    #pragma HLS INLINE
    return x ^ 0x8000; }
inline ap_uint<32> negate_fp32(ap_uint<32> x) { 
    #pragma HLS INLINE
    return x ^ 0x80000000; }

// --- Special Value Checkers ---
inline bool bf16_is_nan(data_t a) { 
    #pragma HLS INLINE
    return ((a & 0x7F80) == 0x7F80) && ((a & 0x007F) != 0); }
inline bool bf16_is_inf(data_t a) { 
    #pragma HLS INLINE
    return ((a & 0x7F80) == 0x7F80) && ((a & 0x007F) == 0); }
inline bool bf16_is_zero(data_t a) { 
    #pragma HLS INLINE
    return ((a & 0x7FFF) == 0x0000); }
inline bool fp32_is_nan(ap_uint<32> a) { 
    #pragma HLS INLINE
    return ((a & 0x7F800000) == 0x7F800000) && ((a & 0x007FFFFF) != 0); }
inline bool fp32_is_inf(ap_uint<32> a) { 
    #pragma HLS INLINE
    return ((a & 0x7F800000) == 0x7F800000) && ((a & 0x007FFFFF) == 0); }

// --- Floating Point Constants ---
const ap_uint<32> FP32_ONE_BITS       = 0x3f800000;
const ap_uint<32> FP32_ZERO_BITS      = 0x00000000;
const ap_uint<32> FP32_HALF_BITS      = 0x3f000000;
const ap_uint<32> FP32_SQRT_1_2_BITS  = 0x3f3504f3;
const data_t BF16_ONE                 = 0x3f80;
const data_t BF16_NEG_ONE             = 0xbf80;
const data_t BF16_ZERO                = 0x0000;
const data_t BF16_NAN                 = 0xffff; // Aligned with golden testbench

// --- Robust Arithmetic Functions ---
inline ap_uint<32> fp32fma(ap_uint<32> a_bits, ap_uint<32> b_bits, ap_uint<32> c_bits) {
    #pragma HLS INLINE
    FloatUIntUnion conv_a = { .u = a_bits }, conv_b = { .u = b_bits }, conv_c = { .u = c_bits };
    float result_float = hls::fma(conv_a.f, conv_b.f, conv_c.f);
    FloatUIntUnion conv_res = { .f = result_float };
    return conv_res.u;
}
inline ap_uint<32> bf16fma(data_t a_in, data_t b_in, data_t c_in) { 
    #pragma HLS INLINE
    return fp32fma(bf16_2_fp32(a_in), bf16_2_fp32(b_in), bf16_2_fp32(c_in)); }
inline ap_uint<32> fp32_add_propagate_specials(ap_uint<32> a, ap_uint<32> b) {
    #pragma HLS PIPELINE II=1
    if (fp32_is_nan(a) || fp32_is_nan(b)) return 0x7FC00000;
    if (fp32_is_inf(a) && fp32_is_inf(b)) {
        if ((a >> 31) != (b >> 31)) return 0x7FC00000;
        else return a;
    }
    if (fp32_is_inf(a)) return a;
    if (fp32_is_inf(b)) return b;
    return fp32fma(a, FP32_ONE_BITS, b);
}
data_t bf16_mul_propagate_specials(data_t a, data_t b); // Declaration
inline data_t bf16_max_propagate_nan(data_t a, data_t b); // Declaration

#endif // ACCELERATOR_UTILS_H