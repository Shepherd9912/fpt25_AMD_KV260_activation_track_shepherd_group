#include "accelerator.h"
#include "hls_math.h"
#include "lut_tables.h"
#include "special_handling_logic.h"
#include "accelerator_utils.h"
// #include "generated_trees.h"
// 高精度 + 资源复用版本
// ===================================================================================
//  SECTION 1 & 2: LOW-LEVEL UTILITIES & ORIGINAL SERIAL PIPELINES
//  !!! NOTE: The functions in this section are deliberately UNCHANGED. !!!
//  They form the "serial islands" that our new parallel architecture will adapt to.
// ===================================================================================
// ===================================================================================
//  NEW: PARALLEL REDUCTION UTILITY FUNCTIONS
// ===================================================================================
inline float add_5(float parameter_0, float parameter_1, float parameter_2, float parameter_3, float parameter_4) {
    float p_sum1, p_sum2;
    p_sum1 = parameter_0 + parameter_1; // Level 1
    p_sum2 = parameter_2 + parameter_3; // Level 1
    p_sum1 = p_sum1 + p_sum2;          // Level 2
    p_sum1 = p_sum1 + parameter_4;     // Level 3 (Final)
    return p_sum1;
}
inline float add_8(float parameter_0, float parameter_1, float parameter_2, float parameter_3, float parameter_4, float parameter_5, float parameter_6, float parameter_7) {
    float p_sum1, p_sum2, p_sum3, p_sum4;
    p_sum1 = parameter_0 + parameter_1; // Level 1
    p_sum2 = parameter_2 + parameter_3; // Level 1
    p_sum3 = parameter_4 + parameter_5; // Level 1
    p_sum4 = parameter_6 + parameter_7; // Level 1
    p_sum1 = p_sum1 + p_sum2;          // Level 2
    p_sum3 = p_sum3 + p_sum4;     // Level 2
    p_sum1 = p_sum1 + p_sum3;     // Level 3 (Final)
    return p_sum1;
}
// ===================================================================================
//  FINAL SOLUTION: Stateful Accumulator Object to enforce II=1
// ===================================================================================
const int FADD_LATENCY = 5;

template<typename T>
struct PipelinedAccumulator {
    // #pragma HLS INLINE
    
    T accums[FADD_LATENCY];

    void init() {
        #pragma HLS INLINE
        #pragma HLS ARRAY_PARTITION variable=accums complete
        INIT_LOOP: for (int i = 0; i < FADD_LATENCY; i++) {
            #pragma HLS UNROLL
            accums[i] = 0.0;
        }
    }

    void accumulate(T val, int index) {
        #pragma HLS INLINE
        // The array partition on `accums` ensures each case accesses a separate register.
        switch(index % FADD_LATENCY) {
            case 0: accums[0] = accums[0] + val; break;
            case 1: accums[1] = accums[1] + val; break;
            case 2: accums[2] = accums[2] + val; break;
            case 3: accums[3] = accums[3] + val; break;
            case 4: accums[4] = accums[4] + val; break;
        }
    }

    T get_sum() {
        #pragma HLS INLINE
        T p_sum1 = accums[0] + accums[1];
        T p_sum2 = accums[2] + accums[3];
        p_sum1 = p_sum1 + p_sum2;
        p_sum1 = p_sum1 + accums[4];
        return p_sum1;
    }
};
void unified_norm_dataflow_core(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode);
template<typename T, int N>
T adder_tree(T data[N]) {
    #pragma HLS INLINE
    T stage[N];
    #pragma HLS ARRAY_PARTITION variable=stage complete dim=1

    // Initialize the leaves of the tree
    INIT_LEAVES: for (int i = 0; i < N; ++i) {
        #pragma HLS UNROLL
        stage[i] = data[i];
    }

    // Iteratively reduce the tree
    REDUCE_STAGES: for (int width = N / 2; width > 0; width /= 2) {
        #pragma HLS UNROLL
        COMBINE_NODES: for (int i = 0; i < width; ++i) {
            #pragma HLS UNROLL
            stage[i] = fp32_add_propagate_specials(stage[i], stage[i + width]);
        }
    }
    return stage[0];
}

template<int N>
data_t max_tree(data_t data[N]) {
    #pragma HLS INLINE
    data_t stage[N];
    #pragma HLS ARRAY_PARTITION variable=stage complete dim=1

    INIT_MAX_LEAVES: for (int i = 0; i < N; ++i) {
        #pragma HLS UNROLL
        stage[i] = data[i];
    }

    REDUCE_MAX_STAGES: for (int width = N / 2; width > 0; width /= 2) {
        #pragma HLS UNROLL
        COMBINE_MAX_NODES: for (int i = 0; i < width; ++i) {
            #pragma HLS UNROLL
            stage[i] = bf16_max_propagate_nan(stage[i], stage[i + width]);
        }
    }
    return stage[0];
}

/**
 * @brief 检查一个bf16值是否可用于初始化max_val的基准。
 *        一个值是“合法”的，当且仅当它既不是NaN，也不是负无穷(-Inf)。
 *        这个逻辑是精确复现原始串行代码行为的关键。
 */
inline bool is_valid_for_max_init(data_t val) {
    #pragma HLS INLINE
    // 0xFF80 is the bit representation of -Inf for bfloat16.
    return !bf16_is_nan(val) && (val != 0xFF80);
}

/**
 * @brief 一个用于存储单个总线（bus）归约结果的“摘要”结构体。
 *        它不仅包含统计数值，还包含关键的元数据(metadata)。
 */
struct BusStats {
    ap_uint<32> sum;
    ap_uint<32> sum_sq;
    data_t      max_val;
    bool        has_valid_max; // 元数据: 指示这个总线的max_val是否“合法”
};

// --- Low-Level Math (Unchanged) ---
data_t bf16_mul_propagate_specials(data_t a, data_t b) {
    #pragma HLS INLINE
    if (bf16_is_nan(a) || bf16_is_nan(b)) return BF16_NAN;
    bool a_is_inf = bf16_is_inf(a); bool b_is_inf = bf16_is_inf(b);
    bool a_is_zero = bf16_is_zero(a); bool b_is_zero = bf16_is_zero(b);
    if ((a_is_inf && b_is_zero) || (b_is_inf && a_is_zero)) return BF16_NAN;
    ap_uint<1> res_sign = (a >> 15) ^ (b >> 15);
    if (a_is_inf || b_is_inf) return res_sign ? 0xFF80 : 0x7F80;
    if (a_is_zero || b_is_zero) return res_sign ? 0x8000 : 0x0000;
    return fp32_2_bf16(bf16fma(a, b, BF16_ZERO));
}
inline data_t bf16_max_propagate_nan(data_t a, data_t b) {
    #pragma HLS INLINE
    if (bf16_is_nan(a)) return b; if (bf16_is_nan(b)) return a;
    ap_uint<1> sign_a = a[15]; ap_uint<1> sign_b = b[15];
    if (sign_a != sign_b) return (sign_b == 1) ? a : b;
    return (sign_a == 0) ? ((a > b) ? a : b) : ((a < b) ? a : b);
}
// [RESTORED] Original function, now correctly handling only EXP and ERF.
// The inaccurate RSQRT/RECIPROCAL logic has been removed.
ap_uint<32> nlu_lookup(data_t query_bf16, NluOp op) {
    #pragma HLS PIPELINE II=1
    #pragma HLS INLINE off
    if (bf16_is_nan(query_bf16)) return bf16_2_fp32(BF16_NAN);
    if (bf16_is_inf(query_bf16)) {
        bool is_neg = query_bf16[15];
        switch(op) {
            case NLU_OP_EXP: return is_neg ? FP32_ZERO_BITS : bf16_2_fp32((data_t)0x7F80);
            case NLU_OP_ERF: return is_neg ? negate_fp32(FP32_ONE_BITS) : FP32_ONE_BITS;
            // RSQRT and RECIPROCAL are now handled by nlu_lookup_fp32
            default: return bf16_2_fp32((data_t)0x7FC0);
        }
    }
    if (bf16_is_zero(query_bf16)) {
        if (op == NLU_OP_EXP) return FP32_ONE_BITS;
        if (op == NLU_OP_ERF) return FP32_ZERO_BITS;
        return FP32_ZERO_BITS; // Should be unreachable for RSQRT/RECIPROCAL
    }

    data_t k_bf16, b_bf16; 
    int index = 0; 
    FloatUIntUnion converter = {0};

    switch (op) {
        case NLU_OP_EXP: {
            converter.u = bf16_2_fp32(query_bf16); float query_f = converter.f;
            if (query_f < -12.0f) return FP32_ZERO_BITS;
            if (query_f < EXP_RANGE_MIN) query_f = EXP_RANGE_MIN; if (query_f > EXP_RANGE_MAX) query_f = EXP_RANGE_MAX;
            index = (int)((query_f - EXP_RANGE_MIN) * EXP_SCALE_FACTOR);
            if (index < 0) index = 0; if (index >= NUM_SEGMENTS) index = NUM_SEGMENTS - 1;
            k_bf16 = EXP_K[index]; b_bf16 = EXP_B[index];
            break;
        }
        case NLU_OP_ERF: {
            converter.u = bf16_2_fp32(query_bf16); float query_f = converter.f;
            if (query_f < 0.0f) query_f = -query_f;
            if (query_f > ERF_RANGE_MAX) query_f = ERF_RANGE_MAX;
            index = (int)((query_f - ERF_RANGE_MIN) * ERF_SCALE_FACTOR);
            if (index < 0) index = 0; if (index >= NUM_SEGMENTS) index = NUM_SEGMENTS - 1;
            k_bf16 = ERF_K[index]; b_bf16 = ERF_B[index];
            break;
        }
        default: // Should not be reached
            k_bf16 = BF16_ZERO; b_bf16 = BF16_ZERO;
            break;
    }

    ap_uint<32> result_fp32 = bf16fma(k_bf16, query_bf16, b_bf16);
    if (op == NLU_OP_ERF) {
        converter.u = bf16_2_fp32(query_bf16);
        if (converter.f < 0.0f) {
            result_fp32 = negate_fp32(result_fp32);
        }
    }
    return result_fp32;
}
ap_uint<32> nlu_lookup_fp32(ap_uint<32> query_fp32, NluOp op) {
    #pragma HLS INLINE

    // --- Special Value Handling ---
    if (fp32_is_nan(query_fp32)) {
        return (ap_uint<32>)0x7FC00000; // Return a quiet NaN
    }

    // Use a union to easily check the sign and value
    FloatUIntUnion converter = { .u = query_fp32 };
    bool is_neg_val = converter.f < 0.0f;

    if (fp32_is_inf(query_fp32)) {
        if (op == NLU_OP_RSQRT) {
            // rsqrt(+inf) -> 0. rsqrt(-inf) -> NaN.
            return is_neg_val ? (ap_uint<32>)0x7FC00000 : FP32_ZERO_BITS;
        } else { // NLU_OP_RECIPROCAL
            // reciprocal(+/-inf) -> +/-0. The sign bit is preserved from the input.
            return is_neg_val ? (ap_uint<32>)0x80000000 : FP32_ZERO_BITS;
        }
    }

    if (converter.f == 0.0f) {
        // rsqrt(+/-0) -> +/-inf. reciprocal(+/-0) -> +/-inf.
        return (query_fp32 & (ap_uint<32>)0x80000000) ? (ap_uint<32>)0xFF800000 : (ap_uint<32>)0x7F800000;
    }
    
    // For rsqrt, any negative input (that is not -inf or -0) is invalid.
    if (op == NLU_OP_RSQRT && is_neg_val) {
        return (ap_uint<32>)0x7FC00000; // Return NaN for sqrt of a negative number
    }

    // --- Original LUT Logic for Normal/Subnormal numbers ---
    ap_uint<8> exp_field = (query_fp32 >> 23) & (ap_uint<32>)0xFF;
    int index;
    
    if (op == NLU_OP_RSQRT) {
        // Handle subnormals by mapping their exponent to the minimum normal exponent for lookup
        if (exp_field == 0) exp_field = RSQRT_MIN_EXP_FIELD; 
        int temp_index = (int)exp_field - HW_EXP_FIELD_START_RSQRT;
        index = (temp_index < 0) ? 0 : ((temp_index >= NUM_SEGMENTS) ? NUM_SEGMENTS - 1 : temp_index);
    } else { // NLU_OP_RECIPROCAL
        if (exp_field == 0) exp_field = RECIPROCAL_MIN_EXP_FIELD;
        int temp_index = (int)exp_field - HW_EXP_FIELD_START_RECIPROCAL;
        index = (temp_index < 0) ? 0 : ((temp_index >= NUM_SEGMENTS) ? NUM_SEGMENTS - 1 : temp_index);
    }

    ap_uint<32> k_fp32, b_fp32;
    if (op == NLU_OP_RSQRT) {
        k_fp32 = RSQRT_K_FP32[index];
        b_fp32 = RSQRT_B_FP32[index];
    } else { 
        k_fp32 = RECIPROCAL_K_FP32[index];
        b_fp32 = RECIPROCAL_B_FP32[index];
    }

    return fp32fma(k_fp32, query_fp32, b_fp32);
}
// [NEW] High-precision lookup function specifically for EXP, using fp32 tables.
ap_uint<32> nlu_lookup_exp_fp32(ap_uint<32> query_fp32) {
    #pragma HLS INLINE
    
    // Use union for easy float manipulation
    FloatUIntUnion converter = { .u = query_fp32 };
    float query_f = converter.f;

    // Clamp the input to the defined range for the LUT
    if (query_f < EXP_RANGE_MIN) query_f = EXP_RANGE_MIN;
    if (query_f > EXP_RANGE_MAX) query_f = EXP_RANGE_MAX;

    // Calculate the index into the LUT
    int index = (int)((query_f - EXP_RANGE_MIN) * EXP_SCALE_FACTOR);
    if (index < 0) index = 0;
    if (index >= NUM_SEGMENTS) index = NUM_SEGMENTS - 1;

    // Fetch the high-precision coefficients
    ap_uint<32> k_fp32 = EXP_K_FP32[index];
    ap_uint<32> b_fp32 = EXP_B_FP32[index];

    // Perform the linear approximation in full fp32 precision
    return fp32fma(k_fp32, query_fp32, b_fp32);
}
/**
 * @brief 使用牛顿-拉弗森法对 rsqrt 的初始猜测值进行一次迭代提纯。
 * @param x_fp32 (x)   需要计算平方根倒数的原始数值 (variance 或 mean_sq)。
 * @param y0_fp32 (y_0) 从LUT中获得的 rsqrt(x) 的高质量初始猜测值。
 * @return y_1         经过一次迭代后精度更高的 rsqrt(x) 结果。
 * @note  迭代公式: y_1 = y_0 * (1.5 - 0.5 * x * y_0^2)
 */
/*inline ap_uint<32> _refine_rsqrt_newton_raphson(ap_uint<32> x_fp32, ap_uint<32> y0_fp32) {
    #pragma HLS INLINE
    
    // 定义牛顿法所需的常数
    const ap_uint<32> FP32_ONE_POINT_FIVE     = 0x3FC00000; // 1.5f
    const ap_uint<32> FP32_NEG_ZERO_POINT_FIVE = 0xBF000000; // -0.5f

    // --- 分步实现迭代公式 y_1 = y_0 * (1.5 - 0.5 * x * y_0^2) ---
    
    // 1. 计算 y_0^2
    ap_uint<32> y0_sq_fp32 = fp32fma(y0_fp32, y0_fp32, FP32_ZERO_BITS);
    
    // 2. 计算括号内的部分 (1.5 - 0.5 * x * y_0^2)，这可以高效地用一个FMA实现
    //    等价于: 1.5 + (-0.5 * x) * y_0^2
    ap_uint<32> neg_half_x_fp32 = fp32fma(x_fp32, FP32_NEG_ZERO_POINT_FIVE, FP32_ZERO_BITS);
    ap_uint<32> inner_term_fp32 = fp32fma(neg_half_x_fp32, y0_sq_fp32, FP32_ONE_POINT_FIVE);

    // 3. 计算最终结果 y_1 = y_0 * inner_term
    ap_uint<32> y1_fp32 = fp32fma(y0_fp32, inner_term_fp32, FP32_ZERO_BITS);

    return y1_fp32;
}*/
// [ULTIMATE-ACCURACY VERSION]
// Refines an initial guess for rsqrt(x) using TWO steps of Newton-Raphson.
// This significantly improves accuracy to meet the final L2 error target.
static inline ap_uint<32> _refine_rsqrt_newton_raphson(ap_uint<32> x_fp32, ap_uint<32> y0_fp32) {
    #pragma HLS INLINE
    
    // Constants for 3.0 and 0.5
    const ap_uint<32> THREE_FP32 = 0x40400000;
    const ap_uint<32> HALF_FP32  = 0x3f000000;

    // --- First Iteration (y0 -> y1) ---
    ap_uint<32> y0_sq   = fp32fma(y0_fp32, y0_fp32, FP32_ZERO_BITS);
    ap_uint<32> term1_1 = fp32fma(x_fp32,  y0_sq,   FP32_ZERO_BITS);
    ap_uint<32> term2_1 = fp32fma(negate_fp32(term1_1), FP32_ONE_BITS, THREE_FP32); // 3.0 - term1_1
    ap_uint<32> term3_1 = fp32fma(y0_fp32, HALF_FP32, FP32_ZERO_BITS); // y0 * 0.5
    ap_uint<32> y1_fp32 = fp32fma(term3_1, term2_1,   FP32_ZERO_BITS); // y1 = (y0 * 0.5) * (3.0 - x*y0^2)

    // --- Second Iteration (y1 -> y2) ---
    ap_uint<32> y1_sq   = fp32fma(y1_fp32, y1_fp32, FP32_ZERO_BITS);
    ap_uint<32> term1_2 = fp32fma(x_fp32,  y1_sq,   FP32_ZERO_BITS);
    ap_uint<32> term2_2 = fp32fma(negate_fp32(term1_2), FP32_ONE_BITS, THREE_FP32); // 3.0 - term1_2
    ap_uint<32> term3_2 = fp32fma(y1_fp32, HALF_FP32, FP32_ZERO_BITS); // y1 * 0.5
    ap_uint<32> y2_fp32 = fp32fma(term3_2, term2_2,   FP32_ZERO_BITS); // y2 = (y1 * 0.5) * (3.0 - x*y1^2)

    return y2_fp32;
}
/**
 * @brief Refines an initial guess for the reciprocal (1/x) using one step of
 *        the Newton-Raphson method.
 * @param x_fp32 The original input value (as fp32 bits).
 * @param y0_fp32 The initial guess for 1/x (as fp32 bits), typically from a LUT.
 * @return A more accurate approximation of 1/x (as fp32 bits).
 * @note Implements the formula: y1 = y0 * (2.0 - x * y0)
 */
/*static inline ap_uint<32> _refine_reciprocal_newton_raphson(ap_uint<32> x_fp32, ap_uint<32> y0_fp32) {
    #pragma HLS INLINE
    
    // Constant for 2.0 in fp32 format
    const ap_uint<32> TWO_FP32 = 0x40000000;

    // Calculate term1 = x * y0
    ap_uint<32> term1 = fp32fma(x_fp32, y0_fp32, FP32_ZERO_BITS);
    
    // Calculate term2 = 2.0 - term1. This is done via fma(term1, -1.0, 2.0)
    // to avoid a separate subtraction step.
    ap_uint<32> neg_term1 = negate_fp32(term1);
    ap_uint<32> term2 = fp32fma(neg_term1, FP32_ONE_BITS, TWO_FP32);

    // Calculate final result y1 = y0 * term2
    ap_uint<32> y1_fp32 = fp32fma(y0_fp32, term2, FP32_ZERO_BITS);
    
    return y1_fp32;
}*/
// [ULTIMATE-ACCURACY VERSION]
// Refines an initial guess for the reciprocal (1/x) using TWO steps of Newton-Raphson.
// This significantly improves accuracy for the softmax calculation.
static inline ap_uint<32> _refine_reciprocal_newton_raphson(ap_uint<32> x_fp32, ap_uint<32> y0_fp32) {
    #pragma HLS INLINE
    
    // Constant for 2.0
    const ap_uint<32> TWO_FP32 = 0x40000000;

    // --- First Iteration (y0 -> y1) ---
    ap_uint<32> term1_1 = fp32fma(x_fp32, y0_fp32, FP32_ZERO_BITS);
    ap_uint<32> term2_1 = fp32fma(negate_fp32(term1_1), FP32_ONE_BITS, TWO_FP32); // 2.0 - term1_1
    ap_uint<32> y1_fp32 = fp32fma(y0_fp32, term2_1, FP32_ZERO_BITS); // y1 = y0 * (2.0 - x*y0)

    // --- Second Iteration (y1 -> y2) ---
    ap_uint<32> term1_2 = fp32fma(x_fp32, y1_fp32, FP32_ZERO_BITS);
    ap_uint<32> term2_2 = fp32fma(negate_fp32(term1_2), FP32_ONE_BITS, TWO_FP32); // 2.0 - term1_2
    ap_uint<32> y2_fp32 = fp32fma(y1_fp32, term2_2, FP32_ZERO_BITS); // y2 = y1 * (2.0 - x*y1)
    
    return y2_fp32;
}
// [MODIFIED] Replaced with a version that uses the optimized reciprocal path
// [FINAL VERSION] Fully corrected SiLU core implementation
ap_uint<32> silu_core_robust(data_t x_bf16) {
    #pragma HLS INLINE
    
    // --- 1. Correctly handle all special cases ---
    if (bf16_is_nan(x_bf16)) return bf16_2_fp32(BF16_NAN);
    if (bf16_is_inf(x_bf16)) {
        bool is_neg = (x_bf16[15] == 1);
        // silu(-inf) = 0 * -inf = NaN
        // silu(+inf) = 1 * +inf = +inf
        return is_neg ? bf16_2_fp32(BF16_NAN) : bf16_2_fp32(x_bf16);
    }

    ap_uint<32> x_fp32 = bf16_2_fp32(x_bf16);
    FloatUIntUnion converter = { .u = x_fp32 };

    // --- 2. Optimization shortcuts for out-of-range values ---
    if (converter.f > 10.0f) return x_fp32;
    if (converter.f < -10.0f) return FP32_ZERO_BITS;

    // --- 3. High-precision core sigmoid calculation ---
    // Calculate -x in fp32
    ap_uint<32> neg_x_fp32 = negate_fp32(x_fp32);

    // Calculate exp(-x) using the new high-precision fp32 lookup
    ap_uint<32> exp_neg_x_fp32 = nlu_lookup_exp_fp32(neg_x_fp32);

    // Calculate 1.0 + exp(-x) in fp32
    ap_uint<32> one_plus_exp_fp32 = fp32_add_propagate_specials(FP32_ONE_BITS, exp_neg_x_fp32);

    // Calculate sigmoid = 1 / (1 + exp(-x)) using our optimized reciprocal path
    ap_uint<32> sigmoid_fp32;
    ap_uint<32> initial_guess = nlu_lookup_fp32(one_plus_exp_fp32, NLU_OP_RECIPROCAL);
    sigmoid_fp32 = _refine_reciprocal_newton_raphson(one_plus_exp_fp32, initial_guess);

    // --- 4. Final multiplication: result = x * sigmoid(x) ---
    return fp32fma(x_fp32, sigmoid_fp32, FP32_ZERO_BITS);
}
// --- Original Serial Helper Functions and Pipelines (Unchanged) ---
struct RowStats { ap_uint<32> sum; ap_uint<32> sum_sq; data_t max_val; };
/**
 * @brief Computes statistics for Norm functions from BRAM.
 * @note  MODIFIED: Now uses the robust "MapReduce" pattern with serial accumulation,
 *        removing the dependency on generated_trees.h.
 */
RowStats reduce_stats_from_bram_norm(data_t bram[D_DIM]) {
    #pragma HLS INLINE
    RowStats stats;
    stats.sum = FP32_ZERO_BITS;
    stats.sum_sq = FP32_ZERO_BITS;
    stats.max_val = BF16_ZERO; // Not used in Norms, but initialize for safety.

    // This loop performs the sequential "Reduce" part across bus results
    REDUCE_STATS_NORM_LOOP: for (int i = 0; i < D_DIM / PARALLEL_FACTOR; ++i) {
        #pragma HLS PIPELINE II=1

        ap_uint<32> local_sum[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_sum complete
        ap_uint<32> local_sum_sq[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_sum_sq complete

        // This loop prepares the data for the parallel "Map" part
        PREPARE_BUS_NORM_DATA: for (int j = 0; j < PARALLEL_FACTOR; ++j) {
            #pragma HLS UNROLL
            data_t val = bram[i * PARALLEL_FACTOR + j];
            if (bf16_is_nan(val)) { val = BF16_ZERO; }
            ap_uint<32> val_fp32 = bf16_2_fp32(val);
            local_sum[j] = val_fp32;
            local_sum_sq[j] = bf16fma(val, val, BF16_ZERO);
        }

        // The parallel "Map" part using the small, robust adder_tree
        ap_uint<32> bus_sum = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum);
        ap_uint<32> bus_sum_sq = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum_sq);

        // Accumulate the results serially
        stats.sum = fp32_add_propagate_specials(stats.sum, bus_sum);
        stats.sum_sq = fp32_add_propagate_specials(stats.sum_sq, bus_sum_sq);
    }
    return stats;
}
// Replace with this BIND_OP definitive version for GUARANTEED II=1.
RowStats reduce_stats_from_bram_regular(data_t bram[D_DIM]) {
    #pragma HLS INLINE

    const int NUM_ITERATIONS = D_DIM / PARALLEL_FACTOR;

    // Phase 1: MAP (Unchanged)
    ap_uint<32> bus_sums_bram[NUM_ITERATIONS];
    ap_uint<32> bus_sums_sq_bram[NUM_ITERATIONS];
    data_t      bus_max_bram[NUM_ITERATIONS];
    #pragma HLS BIND_STORAGE variable=bus_sums_bram type=RAM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=bus_sums_sq_bram type=RAM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=bus_max_bram type=RAM_1P impl=BRAM

    MAP_STATS_LOOP:
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        #pragma HLS PIPELINE II=1
        ap_uint<32> local_sum[PARALLEL_FACTOR], local_sum_sq[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_sum complete
        #pragma HLS ARRAY_PARTITION variable=local_sum_sq complete
        data_t local_max[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_max complete
        PREPARE_BUS_DATA:
        for (int j = 0; j < PARALLEL_FACTOR; ++j) {
            #pragma HLS UNROLL
            data_t val = bram[i * PARALLEL_FACTOR + j];
            ap_uint<32> val_fp32 = bf16_2_fp32(val);
            local_sum[j] = val_fp32;
            local_sum_sq[j] = bf16fma(val, val, BF16_ZERO);
            local_max[j] = val;
        }
        bus_sums_bram[i]    = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum);
        bus_sums_sq_bram[i] = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum_sq);
        bus_max_bram[i]     = max_tree<PARALLEL_FACTOR>(local_max);
    }

    // --- PHASE 2: REDUCE (BIND_OP SYNTHESIS) ---
    RowStats stats;
    stats.max_val = negate_bf16((data_t)0x7F80); // -inf

    /*const int FADD_LATENCY = 5;
    float sum_accums[FADD_LATENCY] = {};
    #pragma HLS ARRAY_PARTITION variable=sum_accums complete
    #pragma HLS BIND_OP variable=sum_accums op=fadd impl=fabric
    REDUCE_SUM_LOOP_II1: for (int i = 0; i < NUM_ITERATIONS; ++i) {
        #pragma HLS PIPELINE II=1
        FloatUIntUnion temp_conv = { .u = bus_sums_bram[i] };
        float val = temp_conv.f;
        switch(i % FADD_LATENCY) {
            case 0: sum_accums[0] = sum_accums[0] + val; break;
            case 1: sum_accums[1] = sum_accums[1] + val; break;
            case 2: sum_accums[2] = sum_accums[2] + val; break;
            case 3: sum_accums[3] = sum_accums[3] + val; break;
            case 4: sum_accums[4] = sum_accums[4] + val; break;
        }
    }
    float final_float_sum = add_5(sum_accums[0], sum_accums[1], sum_accums[2], sum_accums[3], sum_accums[4]);*/
    const int FADD_LATENCY = 8;
    float sum_accums[FADD_LATENCY] = {};
    #pragma HLS ARRAY_PARTITION variable=sum_accums complete
    // #pragma HLS BIND_OP variable=sum_accums op=fadd impl=fabric
    #pragma HLS BIND_OP variable=sum_accums op=fadd impl=dsp
    REDUCE_SUM_LOOP_II1: for (int i = 0; i < NUM_ITERATIONS; ++i) {
        #pragma HLS PIPELINE II=1
        FloatUIntUnion temp_conv = { .u = bus_sums_bram[i] };
        float val = temp_conv.f;
        switch(i % FADD_LATENCY) {
            case 0: sum_accums[0] = sum_accums[0] + val; break;
            case 1: sum_accums[1] = sum_accums[1] + val; break;
            case 2: sum_accums[2] = sum_accums[2] + val; break;
            case 3: sum_accums[3] = sum_accums[3] + val; break;
            case 4: sum_accums[4] = sum_accums[4] + val; break;
            case 5: sum_accums[5] = sum_accums[5] + val; break;
            case 6: sum_accums[6] = sum_accums[6] + val; break;
            case 7: sum_accums[7] = sum_accums[7] + val; break;
        }
    }
    float final_float_sum = add_8(sum_accums[0], sum_accums[1], sum_accums[2], sum_accums[3], sum_accums[4], sum_accums[5], sum_accums[6], sum_accums[7]);
    FloatUIntUnion final_sum_conv = { .f = final_float_sum };
    stats.sum = final_sum_conv.u;

    // --- Accumulator for 'sum_sq' ---
    /*float sum_sq_accums[FADD_LATENCY] = {};
    #pragma HLS ARRAY_PARTITION variable=sum_sq_accums complete
    #pragma HLS BIND_OP variable=sum_sq_accums op=fadd impl=fabric
    REDUCE_SUM_SQ_LOOP_II1: for (int i = 0; i < NUM_ITERATIONS; ++i) {
        #pragma HLS PIPELINE II=1
        FloatUIntUnion temp_conv = { .u = bus_sums_sq_bram[i] };
        float val = temp_conv.f;
        switch(i % FADD_LATENCY) {
            case 0: sum_sq_accums[0] = sum_sq_accums[0] + val; break;
            case 1: sum_sq_accums[1] = sum_sq_accums[1] + val; break;
            case 2: sum_sq_accums[2] = sum_sq_accums[2] + val; break;
            case 3: sum_sq_accums[3] = sum_sq_accums[3] + val; break;
            case 4: sum_sq_accums[4] = sum_sq_accums[4] + val; break;
        }
    }
    float final_float_sum_sq = add_5(sum_sq_accums[0], sum_sq_accums[1], sum_sq_accums[2], sum_sq_accums[3], sum_sq_accums[4]);*/

    float sum_sq_accums[FADD_LATENCY] = {};
    #pragma HLS ARRAY_PARTITION variable=sum_sq_accums complete
    // #pragma HLS BIND_OP variable=sum_sq_accums op=fadd impl=fabric
    #pragma HLS BIND_OP variable=sum_sq_accums op=fadd impl=dsp
    REDUCE_SUM_SQ_LOOP_II1: for (int i = 0; i < NUM_ITERATIONS; ++i) {
        #pragma HLS PIPELINE II=1
        FloatUIntUnion temp_conv = { .u = bus_sums_sq_bram[i] };
        float val = temp_conv.f;
        switch(i % FADD_LATENCY) {
            case 0: sum_sq_accums[0] = sum_sq_accums[0] + val; break;
            case 1: sum_sq_accums[1] = sum_sq_accums[1] + val; break;
            case 2: sum_sq_accums[2] = sum_sq_accums[2] + val; break;
            case 3: sum_sq_accums[3] = sum_sq_accums[3] + val; break;
            case 4: sum_sq_accums[4] = sum_sq_accums[4] + val; break;
            case 5: sum_sq_accums[5] = sum_sq_accums[5] + val; break;
            case 6: sum_sq_accums[6] = sum_sq_accums[6] + val; break;
            case 7: sum_sq_accums[7] = sum_sq_accums[7] + val; break;
        }
    }
    float final_float_sum_sq = add_8(sum_sq_accums[0], sum_sq_accums[1], sum_sq_accums[2], sum_sq_accums[3], sum_sq_accums[4], sum_sq_accums[5], sum_sq_accums[6], sum_sq_accums[7]);

    FloatUIntUnion final_sum_sq_conv = { .f = final_float_sum_sq };
    stats.sum_sq = final_sum_sq_conv.u;

    // --- Max reduction (unchanged) ---
    REDUCE_MAX_LOOP: for (int i = 0; i < NUM_ITERATIONS; ++i) {
        #pragma HLS PIPELINE II=1
        stats.max_val = bf16_max_propagate_nan(stats.max_val, bus_max_bram[i]);
    }

    return stats;
}
template<int D>
void _distribute_bram_to_stream(data_t bram[D], hls::stream<data_t>& out_stream) {
    for (int i = 0; i < D; ++i) { 
        #pragma HLS PIPELINE II=1
        out_stream.write(bram[i]); }
}
void _pass1_buffer_and_stats_norm(hls::stream<data_t>& in, hls::stream<data_t>& data_out, hls::stream<RowStats>& stats_out) {
    #pragma HLS INLINE off
    data_t row_bram[D_DIM];
    #pragma HLS ARRAY_PARTITION variable=row_bram cyclic factor=PARALLEL_FACTOR dim=1

    for (int i = 0; i < D_DIM; ++i) { 
        #pragma HLS PIPELINE II=1
        row_bram[i] = in.read(); 
    }
    stats_out.write(reduce_stats_from_bram_norm(row_bram));
    _distribute_bram_to_stream<D_DIM>(row_bram, data_out);
}
/*void _pass1_buffer_and_stats_regular(hls::stream<data_t>& in, hls::stream<data_t>& data_out, hls::stream<RowStats>& stats_out) {
    #pragma HLS INLINE off
    data_t row_bram[D_DIM];
    #pragma HLS ARRAY_PARTITION variable=row_bram cyclic factor=PARALLEL_FACTOR dim=1

    for (int i = 0; i < D_DIM; ++i) { 
        #pragma HLS PIPELINE II=1
        row_bram[i] = in.read(); 
    }
    stats_out.write(reduce_stats_from_bram_regular(row_bram));
    _distribute_bram_to_stream<D_DIM>(row_bram, data_out);
}*/
// [OPTIMIZED] Replaces the call to the old two-pass reduce function
// with a high-performance single-pass implementation.
void _pass1_buffer_and_stats_regular(
    hls::stream<data_t>& in, 
    hls::stream<data_t>& data_out, 
    hls::stream<RowStats>& stats_out
) {
    #pragma HLS INLINE off
    data_t row_bram[D_DIM];
    #pragma HLS BIND_STORAGE variable=row_bram type=RAM_1P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=row_bram cyclic factor=PARALLEL_FACTOR dim=1

    // Step 1: Buffer the input row into BRAM (Unchanged)
    for (int i = 0; i < D_DIM; ++i) { 
        #pragma HLS PIPELINE II=1
        row_bram[i] = in.read(); 
    }

    // Step 2: High-performance single-pass statistics calculation
    const int NUM_ITERATIONS = D_DIM / PARALLEL_FACTOR;
    const int FADD_LATENCY = 8;
    
    // Pipelined accumulators and max register
    float s_accums[FADD_LATENCY] = {};
    float sq_accums[FADD_LATENCY] = {};
    #pragma HLS ARRAY_PARTITION variable=s_accums complete
    #pragma HLS ARRAY_PARTITION variable=sq_accums complete
    #pragma HLS BIND_OP variable=s_accums op=fadd impl=dsp
    #pragma HLS BIND_OP variable=sq_accums op=fadd impl=dsp
    data_t max_val_reg = negate_bf16((data_t)0x7F80); // Init to -inf

    SINGLE_PASS_STATS_LOOP: for (int i = 0; i < NUM_ITERATIONS; ++i) {
        #pragma HLS PIPELINE II=1 // This loop will now achieve II=1
        
        // --- MAP Part ---
        ap_uint<32> local_sum[PARALLEL_FACTOR], local_sum_sq[PARALLEL_FACTOR];
        data_t local_max[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_sum complete
        #pragma HLS ARRAY_PARTITION variable=local_sum_sq complete
        #pragma HLS ARRAY_PARTITION variable=local_max complete
        
        for (int j = 0; j < PARALLEL_FACTOR; ++j) {
            #pragma HLS UNROLL
            data_t val = row_bram[i * PARALLEL_FACTOR + j];
            ap_uint<32> val_fp32 = bf16_2_fp32(val);
            local_sum[j] = val_fp32;
            local_sum_sq[j] = bf16fma(val, val, BF16_ZERO);
            local_max[j] = val;
        }
        ap_uint<32> bus_sum_fp32 = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum);
        ap_uint<32> bus_sum_sq_fp32 = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum_sq);
        data_t bus_max_bf16 = max_tree<PARALLEL_FACTOR>(local_max);

        // --- REDUCE Part (Concurrent) ---
        FloatUIntUnion temp_s = {.u = bus_sum_fp32};
        float val_s = temp_s.f;
        FloatUIntUnion temp_sq = {.u = bus_sum_sq_fp32};
        float val_sq = temp_sq.f;
        
        int idx = i % FADD_LATENCY;
        s_accums[idx] += val_s;
        sq_accums[idx] += val_sq;
        max_val_reg = bf16_max_propagate_nan(max_val_reg, bus_max_bf16);
    }

    // Final reduction of accumulators
    RowStats stats;
    float final_s_float = add_8(s_accums[0], s_accums[1], s_accums[2], s_accums[3], s_accums[4], s_accums[5], s_accums[6], s_accums[7]);
    FloatUIntUnion final_s_conv = { .f = final_s_float };
    stats.sum = final_s_conv.u;

    float final_sq_float = add_8(sq_accums[0], sq_accums[1], sq_accums[2], sq_accums[3], sq_accums[4], sq_accums[5], sq_accums[6], sq_accums[7]);
    FloatUIntUnion final_sq_conv = { .f = final_sq_float };
    stats.sum_sq = final_sq_conv.u;
    
    stats.max_val = max_val_reg;
    stats_out.write(stats);

    // Step 3: Stream out the buffered data (Unchanged)
    _distribute_bram_to_stream<D_DIM>(row_bram, data_out);
}
// [OPTIMIZED] Replaces the multi-loop, multi-BRAM implementation
// with a more efficient single-pass summation stage.
void _softmax_pass2_normalize_fp32(
    hls::stream<data_t>& data_in, 
    hls::stream<RowStats>& stats_in, 
    hls::stream<ap_uint<32>>& out
) {
    #pragma HLS INLINE off
    RowStats stats = stats_in.read();

    if (bf16_is_nan(stats.max_val)) {
        for(int i=0; i<D_DIM; ++i){ 
            #pragma HLS PIPELINE II=1
            data_in.read(); out.write(bf16_2_fp32(BF16_NAN)); } return;
    }

    // exp_bram is still needed because the final normalization pass needs these values.
    ap_uint<32> exp_bram[D_DIM];
    #pragma HLS BIND_STORAGE variable=exp_bram type=RAM_1P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=exp_bram cyclic factor=PARALLEL_FACTOR dim=1

    // --- Combined Pass 1 (EXP) and Pass 2 (Summation) ---
    const int FADD_LATENCY = 8;
    float sum_accums[FADD_LATENCY] = {};
    #pragma HLS ARRAY_PARTITION variable=sum_accums complete
    #pragma HLS BIND_OP variable=sum_accums op=fadd impl=dsp

    ap_uint<32> neg_max_fp32 = negate_fp32(bf16_2_fp32(stats.max_val));

    // This single loop computes exp, writes it to BRAM for later, and sums it concurrently.
    EXP_AND_SUM_LOOP: for (int i = 0; i < D_DIM / PARALLEL_FACTOR; ++i) {
        #pragma HLS PIPELINE II=1 // This loop will now achieve II=1
        
        // --- MAP Part (calculates bus-level exp sum) ---
        ap_uint<32> local_exp_sum_arr[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_exp_sum_arr complete
        
        for(int j=0; j<PARALLEL_FACTOR; ++j) {
            #pragma HLS UNROLL
            data_t x = data_in.read();
            ap_uint<32> sub_res = fp32_add_propagate_specials(bf16_2_fp32(x), neg_max_fp32);
            ap_uint<32> exp_val = nlu_lookup(fp32_2_bf16(sub_res), NLU_OP_EXP);
            
            exp_bram[i * PARALLEL_FACTOR + j] = exp_val; // Write to BRAM for final pass
            local_exp_sum_arr[j] = fp32_is_nan(exp_val) ? FP32_ZERO_BITS : exp_val;
        }
        ap_uint<32> bus_exp_sum = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_exp_sum_arr);

        // --- REDUCE Part (Concurrent) ---
        FloatUIntUnion temp_conv = { .u = bus_exp_sum };
        float val = temp_conv.f;
        sum_accums[i % FADD_LATENCY] += val;
    }
    // We must drain any remaining inputs if D_DIM is not a multiple of PARALLEL_FACTOR
    // (This example assumes it is, for simplicity. Add drain loop if needed).


    // Final reduction of accumulators
    float final_float_sum = add_8(sum_accums[0], sum_accums[1], sum_accums[2], sum_accums[3], sum_accums[4], sum_accums[5], sum_accums[6], sum_accums[7]);
    FloatUIntUnion final_sum_conv = { .f = final_float_sum };
    ap_uint<32> sum_exp_acc = final_sum_conv.u;

    if (fp32_is_nan(sum_exp_acc)) {
        for (int i = 0; i < D_DIM; ++i) { 
            #pragma HLS PIPELINE II=1
            out.write(bf16_2_fp32(BF16_NAN)); } return;
    }
    FloatUIntUnion sum_check = { .u = sum_exp_acc };
    if (sum_check.f == 0.0f) {
        for (int i = 0; i < D_DIM; ++i) { 
            #pragma HLS PIPELINE II=1
            ap_uint<32> exp_val_fp32 = exp_bram[i]; data_t exp_val_bf16 = fp32_2_bf16(exp_val_fp32);
            data_t res_bf16 = bf16_is_nan(exp_val_bf16) ? BF16_NAN : (bf16_is_zero(exp_val_bf16) ? BF16_NAN : (data_t)0x7F80);
            out.write(bf16_2_fp32(res_bf16));
        } return;
    }

    ap_uint<32> norm_factor_fp32;
    ap_uint<32> initial_guess = nlu_lookup_fp32(sum_exp_acc, NLU_OP_RECIPROCAL);
    if (fp32_is_inf(sum_exp_acc)) {
        norm_factor_fp32 = initial_guess;
    } else {
        norm_factor_fp32 = _refine_reciprocal_newton_raphson(sum_exp_acc, initial_guess);
    }

    NORMALIZE_EXP_LOOP: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        out.write(fp32fma(exp_bram[i], norm_factor_fp32, FP32_ZERO_BITS));
    }
}
// [NEW] A single, unified, and highly optimized core function for all softmax computations.
// This replaces the previous _pass1 and _pass2 functions.
void softmax_compute_core(
    hls::stream<data_t>& in, 
    hls::stream<ap_uint<32>>& out
) {
    #pragma HLS INLINE off

    data_t row_bram[D_DIM];
    #pragma HLS BIND_STORAGE variable=row_bram type=RAM_1P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=row_bram cyclic factor=PARALLEL_FACTOR dim=1
    
    ap_uint<32> exp_bram[D_DIM];
    #pragma HLS BIND_STORAGE variable=exp_bram type=RAM_1P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=exp_bram cyclic factor=PARALLEL_FACTOR dim=1

    // --- Pass 1: Buffer input data and find max_val simultaneously ---
    data_t max_val_reg = negate_bf16((data_t)0x7F80); // Init to -inf
    BUFFER_AND_MAX_LOOP: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        data_t val = in.read();
        row_bram[i] = val;
        max_val_reg = bf16_max_propagate_nan(max_val_reg, val);
    }
    
    if (bf16_is_nan(max_val_reg)) {
        for(int i=0; i<D_DIM; ++i){ 
            #pragma HLS PIPELINE II=1
            out.write(bf16_2_fp32(BF16_NAN)); } return;
    }

    // --- Pass 2: Calculate exp(x - max) and sum them in a single pass ---
    const int FADD_LATENCY = 8;
    float sum_accums[FADD_LATENCY] = {};
    #pragma HLS ARRAY_PARTITION variable=sum_accums complete
    // #pragma HLS BIND_OP variable=sum_accums op=fadd impl=dsp
    #pragma HLS BIND_OP variable=sum_accums op=fadd impl=fabric

    ap_uint<32> neg_max_fp32 = negate_fp32(bf16_2_fp32(max_val_reg));
    const int NUM_ITERATIONS = D_DIM / PARALLEL_FACTOR;

    // This loop reads from BRAM (parallel access) and will achieve II=1
    EXP_AND_SUM_LOOP: for (int i = 0; i < NUM_ITERATIONS; ++i) {
        #pragma HLS PIPELINE II=1
        
        ap_uint<32> local_exp_sum_arr[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_exp_sum_arr complete
        
        // This unrolled loop now reads from BRAM, which supports parallel access
        for(int j=0; j<PARALLEL_FACTOR; ++j) {
            #pragma HLS UNROLL
            data_t x = row_bram[i * PARALLEL_FACTOR + j]; // <-- READ FROM BRAM
            ap_uint<32> sub_res = fp32_add_propagate_specials(bf16_2_fp32(x), neg_max_fp32);
            ap_uint<32> exp_val = nlu_lookup(fp32_2_bf16(sub_res), NLU_OP_EXP);
            
            exp_bram[i * PARALLEL_FACTOR + j] = exp_val;
            local_exp_sum_arr[j] = fp32_is_nan(exp_val) ? FP32_ZERO_BITS : exp_val;
        }
        ap_uint<32> bus_exp_sum = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_exp_sum_arr);

        FloatUIntUnion temp_conv = { .u = bus_exp_sum };
        float val = temp_conv.f;
        sum_accums[i % FADD_LATENCY] += val;
    }

    // Final reduction of sum
    float final_float_sum = add_8(sum_accums[0], sum_accums[1], sum_accums[2], sum_accums[3], sum_accums[4], sum_accums[5], sum_accums[6], sum_accums[7]);
    FloatUIntUnion final_sum_conv = { .f = final_float_sum };
    ap_uint<32> sum_exp_acc = final_sum_conv.u;

    if (fp32_is_nan(sum_exp_acc)) {
        for (int i = 0; i < D_DIM; ++i) { 
            #pragma HLS PIPELINE II=1
            out.write(bf16_2_fp32(BF16_NAN)); } return;
    }
    FloatUIntUnion sum_check = { .u = sum_exp_acc };
    if (sum_check.f == 0.0f) {
        for (int i = 0; i < D_DIM; ++i) { 
            #pragma HLS PIPELINE II=1
            ap_uint<32> exp_val_fp32 = exp_bram[i]; data_t exp_val_bf16 = fp32_2_bf16(exp_val_fp32);
            data_t res_bf16 = bf16_is_nan(exp_val_bf16) ? BF16_NAN : (bf16_is_zero(exp_val_bf16) ? BF16_NAN : (data_t)0x7F80);
            out.write(bf16_2_fp32(res_bf16));
        } return;
    }
    
    // --- Pass 3: Calculate reciprocal and normalize ---
    ap_uint<32> norm_factor_fp32;
    ap_uint<32> initial_guess = nlu_lookup_fp32(sum_exp_acc, NLU_OP_RECIPROCAL);
    if (fp32_is_inf(sum_exp_acc)) {
        norm_factor_fp32 = initial_guess;
    } else {
        norm_factor_fp32 = _refine_reciprocal_newton_raphson(sum_exp_acc, initial_guess);
    }

    NORMALIZE_EXP_LOOP: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        out.write(fp32fma(exp_bram[i], norm_factor_fp32, FP32_ZERO_BITS));
    }
}
// NEW: A struct to hold the statistics result
struct NormStats {
    ap_uint<32> sum;
    ap_uint<32> sum_sq;
};

/**
 * @brief Computes sum and sum_sq from BRAM using the robust "MapReduce" pattern.
 *        This function is based on your proven, working implementation.
 * @param bram The BRAM containing the row data.
 * @return A NormStats struct with the computed sum and sum_sq.
 */
NormStats _unified_norm_stats_from_bram(data_t bram[D_DIM]) {
    #pragma HLS INLINE

    NormStats stats;
    stats.sum = FP32_ZERO_BITS;
    stats.sum_sq = FP32_ZERO_BITS;

    // This loop performs the sequential "Reduce" part across bus results
    STATS_REDUCTION_LOOP: for (int i = 0; i < D_DIM / PARALLEL_FACTOR; ++i) {
        #pragma HLS PIPELINE II=1

        ap_uint<32> local_sum[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_sum complete
        ap_uint<32> local_sum_sq[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_sum_sq complete

        // This loop prepares the data for the parallel "Map" part
        PREPARE_BUS_DATA: for (int j = 0; j < PARALLEL_FACTOR; ++j) {
            #pragma HLS UNROLL
            data_t val = bram[i * PARALLEL_FACTOR + j];
            ap_uint<32> val_fp32 = bf16_is_nan(val) ? FP32_ZERO_BITS : bf16_2_fp32(val);
            local_sum[j] = val_fp32;
            local_sum_sq[j] = fp32fma(val_fp32, val_fp32, FP32_ZERO_BITS);
        }

        // The parallel "Map" part using the small, robust adder_tree
        ap_uint<32> bus_sum = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum);
        ap_uint<32> bus_sum_sq = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum_sq);

        // Accumulate the results
        stats.sum = fp32_add_propagate_specials(stats.sum, bus_sum);
        stats.sum_sq = fp32_add_propagate_specials(stats.sum_sq, bus_sum_sq);
    }
    return stats;
}
// [FINAL OPTIMIZED VERSION - FLOAT ACCUMULATORS]
// Replaces the two-pass MapReduce with a single-pass implementation to
// eliminate intermediate BRAMs and resolve the II=7 performance bottleneck.
// This version uses standard 'float' accumulators as requested.
void _unified_norm_compute_core(
    hls::stream<data_t>& in,
    hls::stream<ap_uint<32>>& out,
    ActivationOpcode opcode
) {
    #pragma HLS INLINE off
    data_t row_bram[D_DIM];
    #pragma HLS BIND_STORAGE variable=row_bram type=RAM_1P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=row_bram cyclic factor=PARALLEL_FACTOR dim=1
    
    // --- Phase 1: Buffer data into BRAM (Unchanged) ---
    UNIFIED_BUFFER_LOOP: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        row_bram[i] = in.read();
    }

    // --- Phase 2: Single-Pass Map & Reduce (KEY OPTIMIZATION) ---
    const int NUM_ITERATIONS = D_DIM / PARALLEL_FACTOR;
    const int FADD_LATENCY = 8; // Latency for float add can also be 7-8
    ap_uint<32> sum, sum_sq;

    // Standard float accumulators
    float s_accums[FADD_LATENCY] = {};
    float sq_accums[FADD_LATENCY] = {};
    #pragma HLS ARRAY_PARTITION variable=s_accums complete
    #pragma HLS ARRAY_PARTITION variable=sq_accums complete
    // #pragma HLS BIND_OP variable=s_accums op=fadd impl=dsp
    // #pragma HLS BIND_OP variable=sq_accums op=fadd impl=dsp
    #pragma HLS BIND_OP variable=s_accums op=fadd impl=fabric
    #pragma HLS BIND_OP variable=sq_accums op=fadd impl=fabric

    // This single loop now performs both MAP and REDUCE concurrently.
    SINGLE_PASS_MAP_REDUCE: for (int i = 0; i < NUM_ITERATIONS; ++i) {
        #pragma HLS PIPELINE II=1 // This loop should now achieve II=1

        // --- MAP part (calculates stats for one bus-width chunk) ---
        ap_uint<32> local_sum[PARALLEL_FACTOR], local_sum_sq[PARALLEL_FACTOR];
        #pragma HLS ARRAY_PARTITION variable=local_sum complete
        #pragma HLS ARRAY_PARTITION variable=local_sum_sq complete
        for (int j = 0; j < PARALLEL_FACTOR; ++j) {
            #pragma HLS UNROLL
            data_t val = row_bram[i * PARALLEL_FACTOR + j];
            ap_uint<32> val_fp32 = bf16_is_nan(val) ? FP32_ZERO_BITS : bf16_2_fp32(val);
            local_sum[j] = val_fp32;
            local_sum_sq[j] = fp32fma(val_fp32, val_fp32, FP32_ZERO_BITS);
        }
        ap_uint<32> bus_sum_fp32 = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum);
        ap_uint<32> bus_sum_sq_fp32 = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sum_sq);

        // --- REDUCE part (immediately accumulates the bus stats into float accumulators) ---
        FloatUIntUnion temp_s = {.u = bus_sum_fp32};
        float val_s = temp_s.f;
        
        FloatUIntUnion temp_sq = {.u = bus_sum_sq_fp32};
        float val_sq = temp_sq.f;

        // Modulo indexing breaks the dependency chain for the accumulator
        int idx = i % FADD_LATENCY;
        s_accums[idx] = s_accums[idx] + val_s;
        sq_accums[idx] = sq_accums[idx] + val_sq;
    }

    // Final reduction of the accumulator array (using float version of add_8)
    float final_s_float = add_8(s_accums[0], s_accums[1], s_accums[2], s_accums[3], s_accums[4], s_accums[5], s_accums[6], s_accums[7]);
    FloatUIntUnion final_s_conv = { .f = final_s_float };
    sum = final_s_conv.u;

    float final_sq_float = add_8(sq_accums[0], sq_accums[1], sq_accums[2], sq_accums[3], sq_accums[4], sq_accums[5], sq_accums[6], sq_accums[7]);
    FloatUIntUnion final_sq_conv = { .f = final_sq_float };
    sum_sq = final_sq_conv.u;

    // --- Phase 3: Calculate Norm Factor and Apply ---
    // The rest of this function is identical to your last working version.
    // It will now receive the `sum` and `sum_sq` values much faster.
    ap_uint<32> norm_factor_fp32;
    ap_uint<32> mean_fp32 = FP32_ZERO_BITS;
    const double INV_D_DIM_DBL = 1.0 / (double)D_DIM;
    FloatUIntUnion inv_count_u = {0}; 
    inv_count_u.f = INV_D_DIM_DBL;

    if (opcode == OP_LAYERNORM) {
        mean_fp32 = fp32fma(sum, inv_count_u.u, FP32_ZERO_BITS);
        ap_uint<32> neg_mean_fp32 = negate_fp32(mean_fp32);
        
        // This second MapReduce for variance also needs to be single-pass
        float d_accums[FADD_LATENCY] = {};
        #pragma HLS ARRAY_PARTITION variable=d_accums complete
        // #pragma HLS BIND_OP variable=d_accums op=fadd impl=dsp
        #pragma HLS BIND_OP variable=d_accums op=fadd impl=fabric
        
        UNIFIED_PASS2_VAR_MAP_REDUCE: for (int i = 0; i < NUM_ITERATIONS; ++i) {
            #pragma HLS PIPELINE II=1
            
            ap_uint<32> local_sq_diff[PARALLEL_FACTOR];
            #pragma HLS ARRAY_PARTITION variable=local_sq_diff complete
            for (int j = 0; j < PARALLEL_FACTOR; ++j) {
                #pragma HLS UNROLL
                data_t val_bf16 = row_bram[i * PARALLEL_FACTOR + j];
                ap_uint<32> centered_sq = FP32_ZERO_BITS;
                if (!bf16_is_nan(val_bf16)) {
                    ap_uint<32> val_fp32 = bf16_2_fp32(val_bf16);
                    ap_uint<32> centered = fp32_add_propagate_specials(val_fp32, neg_mean_fp32);
                    centered_sq = fp32fma(centered, centered, FP32_ZERO_BITS);
                }
                local_sq_diff[j] = centered_sq;
            }
            ap_uint<32> bus_sq_diff = adder_tree<ap_uint<32>, PARALLEL_FACTOR>(local_sq_diff);
            
            FloatUIntUnion temp_d = {.u = bus_sq_diff};
            float val_d = temp_d.f;
            d_accums[i % FADD_LATENCY] += val_d;
        }

        float final_d_float = add_8(d_accums[0], d_accums[1], d_accums[2], d_accums[3], d_accums[4], d_accums[5], d_accums[6], d_accums[7]);
        FloatUIntUnion final_d_conv = { .f = final_d_float };
        ap_uint<32> sum_sq_diff = final_d_conv.u;
    
        ap_uint<32> var_fp32 = fp32fma(sum_sq_diff, inv_count_u.u, FP32_ZERO_BITS);
        FloatUIntUnion var_u = {.u = var_fp32};
        if (var_u.f < 0.0f) var_u.f = 0.0f; 
        var_u.f += 1e-5f;

        ap_uint<32> initial_guess = nlu_lookup_fp32(var_u.u, NLU_OP_RSQRT);
        if (fp32_is_inf(var_u.u) || var_u.f == 0.0f) {
            norm_factor_fp32 = initial_guess;
        } else {
            norm_factor_fp32 = _refine_rsqrt_newton_raphson(var_u.u, initial_guess);
        }
    
    } else { // OP_RMSNORM
        ap_uint<32> mean_sq = fp32fma(sum_sq, inv_count_u.u, FP32_ZERO_BITS);
        FloatUIntUnion mean_sq_u = {.u = mean_sq};
        if (mean_sq_u.f < 0.0f) mean_sq_u.f = 0.0f; 
        mean_sq_u.f += 1e-5f;
        
        ap_uint<32> initial_guess = nlu_lookup_fp32(mean_sq_u.u, NLU_OP_RSQRT);
        if (fp32_is_inf(mean_sq_u.u) || mean_sq_u.f == 0.0f) {
            norm_factor_fp32 = initial_guess;
        } else {
            norm_factor_fp32 = _refine_rsqrt_newton_raphson(mean_sq_u.u, initial_guess);
        }
    }
    
    // Pass 3 - Apply normalization (Unchanged)
    ap_uint<32> neg_mean_fp32 = negate_fp32(mean_fp32);
    UNIFIED_PASS3_NORM: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        ap_uint<32> val_fp32 = bf16_2_fp32(row_bram[i]);
        ap_uint<32> val_to_norm = (opcode == OP_LAYERNORM) ?
                                  fp32_add_propagate_specials(val_fp32, neg_mean_fp32) :
                                  val_fp32;
        ap_uint<32> normalized_fp32 = fp32fma(val_to_norm, norm_factor_fp32, FP32_ZERO_BITS);
        out.write(normalized_fp32);
    }
}
void _rmsnorm_pass2_normalize_fp32(hls::stream<data_t>& data_in, hls::stream<RowStats>& stats_in, hls::stream<ap_uint<32>>& out) {
    #pragma HLS INLINE off
    RowStats stats = stats_in.read();
    
    // Constant for 1.0 / D_DIM in fp32
    FloatUIntUnion inv_count_u = {0}; inv_count_u.f = 1.0f / (float)D_DIM;
    
    ap_uint<32> mean_sq = fp32fma(stats.sum_sq, inv_count_u.u, FP32_ZERO_BITS);
    FloatUIntUnion mean_sq_u = {.u = mean_sq};
    
    if (mean_sq_u.f < 0.0f) mean_sq_u.f = 0.0f;
    mean_sq_u.f += 1e-5f; // Add epsilon
    
    FloatUIntUnion rsqrt_u = {0}; rsqrt_u.f = hls::rsqrt(mean_sq_u.f);
    // NEW: Keep normalization factor in high precision.
    ap_uint<32> norm_factor_fp32 = rsqrt_u.u;
    
    for(int i=0; i<D_DIM; ++i){
        #pragma HLS PIPELINE II=1
        data_t x = data_in.read();
        // NEW: Final multiplication is done in fp32.
        out.write(fp32fma(bf16_2_fp32(x), norm_factor_fp32, FP32_ZERO_BITS));
    }
}

// --- Finalize functions now accept fp32 stream and perform the final cast ---

void _softmax_finalize_row_fp32(RowCharacteristics d, hls::stream<ap_uint<32>>& reg_in, hls::stream<data_t>& out) {
    #pragma HLS INLINE off
    switch (d.code) {
        case OUTPUT_ALL_NAN:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                reg_in.read();
                out.write(BF16_NAN);
            }
            break;

        case PROCESS_NORMALLY:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                // NEW: Read from fp32 stream and cast to bf16 at the very end.
                out.write(fp32_2_bf16(reg_in.read()));
            }
            break;

        default: 
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                reg_in.read();
                out.write(BF16_NAN);
            }
            break;
    }
}


void _rmsnorm_finalize_row_fp32(RowCharacteristics d, hls::stream<ap_uint<32>>& reg_in, hls::stream<data_t>& out) {
    #pragma HLS INLINE off
    switch (d.code) {
        case NORM_CONST_OVERFLOW:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                reg_in.read();
                out.write(BF16_ZERO);
            }
            break;
        
        case OUTPUT_ALL_NAN:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                reg_in.read();
                out.write(BF16_NAN);
            }
            break;

        case NORM_CONST_REGULAR:
        case PROCESS_NORMALLY:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                // NEW: Read from fp32 stream and cast to bf16 at the very end.
                out.write(fp32_2_bf16(reg_in.read()));
            }
            break;

        default:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                reg_in.read(); out.write(BF16_NAN);
            }
            break;
    }
}

void _distribute_row_to_streams_2(hls::stream<data_t>& in, hls::stream<data_t>& out1, hls::stream<data_t>& out2) {
    #pragma HLS INLINE off
    for (int i = 0; i < D_DIM; ++i) { 
        #pragma HLS PIPELINE II=1
        data_t val = in.read(); out1.write(val); out2.write(val); }
}

// --- MODIFIED ROW CORES to use the new fp32 pipelines ---

/*void softmax_row_core(hls::stream<data_t>& in, hls::stream<data_t>& out) {
    #pragma HLS DATAFLOW
    hls::stream<data_t> analyzer_in("analyzer_in_sm_core");
    hls::stream<data_t> regular_in("regular_in_sm_core");
    hls::stream<RowStats> stats_stream("stats_stream_sm_core");
    hls::stream<data_t> data_for_pass2("data_for_pass2_sm_core");
    // NEW: Intermediate stream for high-precision results.
    hls::stream<ap_uint<32>> regular_out_fp32("regular_out_fp32_sm_core");

    #pragma HLS STREAM variable=analyzer_in       depth=D_DIM
    #pragma HLS STREAM variable=regular_in        depth=D_DIM
    #pragma HLS STREAM variable=stats_stream      depth=2
    #pragma HLS STREAM variable=data_for_pass2    depth=D_DIM
    #pragma HLS STREAM variable=regular_out_fp32  depth=D_DIM

    _distribute_row_to_streams_2(in, analyzer_in, regular_in);
    RowCharacteristics decision = _analyze_row_characteristics(analyzer_in, OP_SOFTMAX);
    _pass1_buffer_and_stats_regular(regular_in, data_for_pass2, stats_stream);
    // MODIFIED: Call the new fp32 version of pass2.
    _softmax_pass2_normalize_fp32(data_for_pass2, stats_stream, regular_out_fp32);
    // MODIFIED: Call the finalize function that accepts fp32 stream.
    _softmax_finalize_row_fp32(decision, regular_out_fp32, out);
}*/
// [MODIFIED] Simplified row core that dispatches to the new unified compute function
void softmax_row_core(hls::stream<data_t>& in, hls::stream<data_t>& out) {
    #pragma HLS DATAFLOW
    hls::stream<data_t> analyzer_in("analyzer_in_sm_core");
    hls::stream<data_t> regular_in("regular_in_sm_core");
    hls::stream<ap_uint<32>> regular_out_fp32("regular_out_fp32_sm_core");

    #pragma HLS STREAM variable=analyzer_in       depth=D_DIM
    #pragma HLS STREAM variable=regular_in        depth=D_DIM
    #pragma HLS STREAM variable=regular_out_fp32  depth=D_DIM

    // Distribute input for analysis and computation
    _distribute_row_to_streams_2(in, analyzer_in, regular_in);
    
    // Analyze for special cases (e.g., +/-inf together)
    RowCharacteristics decision = _analyze_row_characteristics(analyzer_in, OP_SOFTMAX);
    
    // Call the single, unified compute core for the regular path
    softmax_compute_core(regular_in, regular_out_fp32);
    
    // Finalize the output based on the initial analysis
    _softmax_finalize_row_fp32(decision, regular_out_fp32, out);
}
// split the layernorm into 2~3 functions
/**
 * @brief LayerNorm Pass 1: Buffers data, calculates mean, and streams data out.
 * @param in          Input data stream (bf16).
 * @param data_out    Output data stream (bf16), a copy of the input.
 * @param stats_out   Output stream for statistics, containing only the mean (fp32).
 */
void _layernorm_pass1_buffer_and_stats(
    hls::stream<data_t>& in,
    hls::stream<data_t>& data_out,
    hls::stream<ap_uint<32>>& stats_out
) {
    #pragma HLS INLINE off
    data_t row_bram[D_DIM];
    #pragma HLS BIND_STORAGE variable=row_bram type=RAM_1P impl=BRAM

    ap_uint<32> sum = FP32_ZERO_BITS;

    // Buffer the row and calculate the sum
    BUFFER_AND_SUM: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        data_t val = in.read();
        row_bram[i] = val;
        // NaNs are treated as zero for the mean calculation
        sum = fp32_add_propagate_specials(sum, bf16_is_nan(val) ? FP32_ZERO_BITS : bf16_2_fp32(val));
    }

    // Calculate mean and write it to the stats stream
    FloatUIntUnion inv_count_u = {0}; 
    inv_count_u.f = 1.0f / (float)D_DIM;
    ap_uint<32> mean_fp32 = fp32fma(sum, inv_count_u.u, FP32_ZERO_BITS);
    stats_out.write(mean_fp32);

    // Stream the buffered data out for the next pass
    STREAM_OUT: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        data_out.write(row_bram[i]);
    }
}
/**
 * @brief LayerNorm Pass 2: Calculates variance and normalizes using a stable
 *        two-pass approach internally.
 * @param data_in     Input data stream (bf16).
 * @param stats_in    Input stream containing the pre-calculated mean (fp32).
 * @param out         Output data stream with normalized values (fp32).
 */
void _layernorm_pass2_normalize_stable(
    hls::stream<data_t>& data_in,
    hls::stream<ap_uint<32>>& stats_in,
    hls::stream<ap_uint<32>>& out
) {
    #pragma HLS INLINE off
    // Read the mean calculated in Pass 1
    ap_uint<32> mean_fp32 = stats_in.read();
    ap_uint<32> neg_mean_fp32 = negate_fp32(mean_fp32);

    data_t row_bram[D_DIM];
    #pragma HLS BIND_STORAGE variable=row_bram type=RAM_1P impl=BRAM

    // Internal Pass A: Buffer data and calculate sum of squared differences
    ap_uint<32> sum_sq_diff = FP32_ZERO_BITS;
    PASS2A_BUFFER_AND_VAR: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        data_t val = data_in.read();
        row_bram[i] = val;
        ap_uint<32> val_fp32 = bf16_2_fp32(val);
        ap_uint<32> centered = fp32_add_propagate_specials(val_fp32, neg_mean_fp32);
        // NaNs do not contribute to variance.
        ap_uint<32> centered_sq = fp32fma(centered, centered, FP32_ZERO_BITS);
        sum_sq_diff = fp32_add_propagate_specials(sum_sq_diff, bf16_is_nan(val) ? FP32_ZERO_BITS : centered_sq);
    }

    // Calculate variance and the normalization factor
    FloatUIntUnion inv_count_u = {0}; 
    inv_count_u.f = 1.0f / (float)D_DIM;
    ap_uint<32> var_fp32 = fp32fma(sum_sq_diff, inv_count_u.u, FP32_ZERO_BITS);
    
    FloatUIntUnion var_u = {.u = var_fp32};
    if (var_u.f < 0.0f) var_u.f = 0.0f;
    var_u.f += 1e-5f; // Epsilon
    
    FloatUIntUnion rsqrt_u = {0}; 
    rsqrt_u.f = hls::rsqrt(var_u.f);
    ap_uint<32> norm_factor_fp32 = rsqrt_u.u;

    // Internal Pass B: Normalize the buffered data and stream out
    PASS2B_NORMALIZE: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        data_t x = row_bram[i];
        ap_uint<32> centered = fp32_add_propagate_specials(bf16_2_fp32(x), neg_mean_fp32);
        ap_uint<32> normalized_fp32 = fp32fma(centered, norm_factor_fp32, FP32_ZERO_BITS);
        out.write(normalized_fp32);
    }
}
/**
 * @brief LayerNorm Finalize: Selects between regular path (fp32) and special
 *        case outputs, performing the final cast to bf16.
 */
void _layernorm_finalize_row_fp32(
    RowCharacteristics d,
    hls::stream<ap_uint<32>>& reg_in,
    hls::stream<data_t>& out
) {
    #pragma HLS INLINE off
    switch (d.code) {
        case NORM_CONST_OVERFLOW:
        case OUTPUT_ALL_NAN:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                reg_in.read(); 
                out.write(BF16_NAN);
            }
            break;

        case NORM_CONST_REGULAR:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                reg_in.read();
                out.write(BF16_ZERO);
            }
            break;

        case PROCESS_NORMALLY:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                out.write(fp32_2_bf16(reg_in.read()));
            }
            break;

        default:
            for (int i=0; i<D_DIM; ++i) {
                #pragma HLS PIPELINE II=1
                reg_in.read(); 
                out.write(BF16_NAN);
            }
            break;
    }
}
// ===================================================================================
// NEW: UNIFIED LAYERNORM CORE
// Re-written to use a numerically stable two-pass algorithm.
// This replaces the old dataflow structure for LayerNorm.
// ===================================================================================
void layernorm_row_core_unified(hls::stream<data_t>& in, RowCharacteristics& decision, hls::stream<data_t>& out) {
    #pragma HLS INLINE off
    
    // Handle special cases first, as they don't need the BRAM.
    if (decision.code != PROCESS_NORMALLY) {
        for (int i = 0; i < D_DIM; ++i) {
            #pragma HLS PIPELINE II=1
            in.read(); // Consume input
            if (decision.code == NORM_CONST_REGULAR) {
                out.write(BF16_ZERO);
            } else { // NORM_CONST_OVERFLOW or OUTPUT_ALL_NAN
                out.write(BF16_NAN);
            }
        }
        return;
    }

    // --- PROCESS_NORMALLY case ---
    data_t row_bram[D_DIM];
    #pragma HLS BIND_STORAGE variable=row_bram type=RAM_1P impl=BRAM

    ap_uint<32> sum = FP32_ZERO_BITS;

    // PASS 1: Buffer row and calculate sum
    PASS1_LN_MEAN: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        data_t val = in.read();
        row_bram[i] = val;
        ap_uint<32> val_fp32 = bf16_2_fp32(val);
        // For LayerNorm, NaNs are treated as zero in the sum.
        sum = fp32_add_propagate_specials(sum, bf16_is_nan(val) ? FP32_ZERO_BITS : val_fp32);
    }
    
    // Calculate mean
    FloatUIntUnion inv_count_u = {0}; 
    inv_count_u.f = 1.0f / (float)D_DIM;
    ap_uint<32> mean_fp32 = fp32fma(sum, inv_count_u.u, FP32_ZERO_BITS);
    ap_uint<32> neg_mean_fp32 = negate_fp32(mean_fp32);

    // PASS 2: Calculate sum of squared differences from the mean
    ap_uint<32> sum_sq_diff = FP32_ZERO_BITS;
    PASS2_LN_VAR: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        data_t val = row_bram[i];
        ap_uint<32> val_fp32 = bf16_2_fp32(val);
        ap_uint<32> centered = fp32_add_propagate_specials(val_fp32, neg_mean_fp32);
        // NaNs do not contribute to variance.
        sum_sq_diff = fp32fma(centered, centered, bf16_is_nan(val) ? sum_sq_diff : fp32_add_propagate_specials(sum_sq_diff, FP32_ZERO_BITS));
    }
    
    // Calculate variance and normalization factor
    ap_uint<32> var_fp32 = fp32fma(sum_sq_diff, inv_count_u.u, FP32_ZERO_BITS);
    FloatUIntUnion var_u = {.u = var_fp32};
    if (var_u.f < 0.0f) var_u.f = 0.0f;
    var_u.f += 1e-5f; // Epsilon
    
    FloatUIntUnion rsqrt_u = {0}; 
    rsqrt_u.f = hls::rsqrt(var_u.f);
    ap_uint<32> norm_factor_fp32 = rsqrt_u.u;
    
    // PASS 3: Normalize and write output
    PASS3_LN_NORM: for (int i = 0; i < D_DIM; ++i) {
        #pragma HLS PIPELINE II=1
        data_t x = row_bram[i];
        ap_uint<32> centered = fp32_add_propagate_specials(bf16_2_fp32(x), neg_mean_fp32);
        ap_uint<32> normalized_fp32 = fp32fma(centered, norm_factor_fp32, FP32_ZERO_BITS);
        out.write(fp32_2_bf16(normalized_fp32));
    }
}

// reused layernorm core:
void layernorm_row_core(hls::stream<data_t>& in, hls::stream<data_t>& out) {
    #pragma HLS DATAFLOW
    hls::stream<data_t> analyzer_in("analyzer_in_ln_core");
    hls::stream<data_t> regular_in("regular_in_ln_core");
    hls::stream<ap_uint<32>> regular_out_fp32("regular_out_fp32_ln_core");
    #pragma HLS STREAM variable=analyzer_in       depth=D_DIM
    #pragma HLS STREAM variable=regular_in        depth=D_DIM
    #pragma HLS STREAM variable=regular_out_fp32  depth=D_DIM

    // Dataflow2 (Analyzer) - Stays specific to LayerNorm
    _distribute_row_to_streams_2(in, analyzer_in, regular_in);
    RowCharacteristics decision = _analyze_row_characteristics(analyzer_in, OP_LAYERNORM);

    // Dataflow1 (Compute) - Calls the new unified core
    _unified_norm_compute_core(regular_in, regular_out_fp32, OP_LAYERNORM);

    // Dataflow2 (Finalizer) - Stays specific to LayerNorm
    _layernorm_finalize_row_fp32(decision, regular_out_fp32, out);
}
// reused rmsnorm core:
void rmsnorm_row_core(hls::stream<data_t>& in, hls::stream<data_t>& out) {
    #pragma HLS DATAFLOW
    hls::stream<data_t> analyzer_in("analyzer_in_rms_core");
    hls::stream<data_t> regular_in("regular_in_rms_core");
    hls::stream<ap_uint<32>> regular_out_fp32("regular_out_fp32_rms_core");
    #pragma HLS STREAM variable=analyzer_in       depth=D_DIM
    #pragma HLS STREAM variable=regular_in        depth=D_DIM
    #pragma HLS STREAM variable=regular_out_fp32  depth=D_DIM

    // Dataflow2 (Analyzer) - Stays specific to RMSNorm
    _distribute_row_to_streams_2(in, analyzer_in, regular_in);
    RowCharacteristics decision = _analyze_row_characteristics(analyzer_in, OP_RMSNORM);

    // Dataflow1 (Compute) - Calls the new unified core
    _unified_norm_compute_core(regular_in, regular_out_fp32, OP_RMSNORM);

    // Dataflow2 (Finalizer) - Stays specific to RMSNorm
    _rmsnorm_finalize_row_fp32(decision, regular_out_fp32, out);
}

// --- TOP-LEVEL PIPELINES (Largely unchanged, they just call the modified row_cores) ---

void softmax_pipeline(hls::stream<data_t>& in, hls::stream<data_t>& out, ActivationOpcode opcode) {
    hls::stream<data_t> in_row("in_row_sm");
    hls::stream<data_t> out_row("out_row_sm");
    #pragma HLS STREAM variable=in_row  depth=D_DIM
    #pragma HLS STREAM variable=out_row depth=D_DIM

    for (int n = 0; n < N_DIM; ++n) {
        #pragma HLS LOOP_FLATTEN off
        
        for(int i = 0; i < D_DIM; ++i) {
            #pragma HLS PIPELINE
            in_row.write(in.read());
        }
        softmax_row_core(in_row, out_row);
        for(int i = 0; i < D_DIM; ++i) {
            #pragma HLS PIPELINE
            out.write(out_row.read());
        }
    }
}

void layernorm_pipeline(hls::stream<data_t>& in, hls::stream<data_t>& out, ActivationOpcode opcode) {
    hls::stream<data_t> in_row("in_row_ln");
    hls::stream<data_t> out_row("out_row_ln");
    #pragma HLS STREAM variable=in_row  depth=D_DIM
    #pragma HLS STREAM variable=out_row depth=D_DIM

    for (int n = 0; n < N_DIM; ++n) {
        #pragma HLS LOOP_FLATTEN off

        for(int i = 0; i < D_DIM; ++i) {
            #pragma HLS PIPELINE
            in_row.write(in.read());
        }
        layernorm_row_core(in_row, out_row); // This now calls the new unified core
        for(int i = 0; i < D_DIM; ++i) {
            #pragma HLS PIPELINE
            out.write(out_row.read());
        }
    }
}

void rmsnorm_pipeline(hls::stream<data_t>& in, hls::stream<data_t>& out, ActivationOpcode opcode) {
    hls::stream<data_t> in_row("in_row_rms");
    hls::stream<data_t> out_row("out_row_rms");
    #pragma HLS STREAM variable=in_row  depth=D_DIM
    #pragma HLS STREAM variable=out_row depth=D_DIM

    for (int n = 0; n < N_DIM; ++n) {
        #pragma HLS LOOP_FLATTEN off

        for(int i = 0; i < D_DIM; ++i) {
            #pragma HLS PIPELINE
            in_row.write(in.read());
        }
        rmsnorm_row_core(in_row, out_row);
        for(int i = 0; i < D_DIM; ++i) {
            #pragma HLS PIPELINE
            out.write(out_row.read());
        }
    }
}

// ===================================================================================
//  UNCHANGED SECTIONS: Elementwise, Nonlinear, Adapters, Scheduler
// ===================================================================================
void elementwise_pipeline(hls::stream<WideBus>& in_a, hls::stream<WideBus>& in_b, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    const int total_buses = (N_DIM * D_DIM) / PARALLEL_FACTOR;
    if (opcode == OP_ADD || opcode == OP_MUL) {
        COMPUTE_ELEMENTWISE: for (int i = 0; i < total_buses; ++i) {
            #pragma HLS PIPELINE II=1
            WideBus bus_a = in_a.read();
            WideBus bus_b = in_b.read();
            WideBus bus_out;
            PROCESS_ELEMENTWISE_BUS: for (int j = 0; j < PARALLEL_FACTOR; ++j) {
                #pragma HLS UNROLL
                data_t a = bus_a.data[j];
                data_t b = bus_b.data[j];
                if (opcode == OP_ADD) {
                    bus_out.data[j] = fp32_2_bf16(fp32_add_propagate_specials(bf16_2_fp32(a), bf16_2_fp32(b)));
                } else {
                    bus_out.data[j] = bf16_mul_propagate_specials(a, b);
                }
            }
            out.write(bus_out);
        }
    } else {
        DRAIN_ELEMENTWISE: for (int i = 0; i < total_buses; ++i) {
            #pragma HLS PIPELINE II=1
            in_a.read(); in_b.read(); out.write(WideBus());
        }
    }
}
/*void nonlinear_pipeline(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    const int total_buses = (N_DIM * D_DIM) / PARALLEL_FACTOR;
    if (opcode == OP_SILU || opcode == OP_GELU) {
        COMPUTE_NONLINEAR: for (int i = 0; i < total_buses; ++i) {
            #pragma HLS PIPELINE II=1
            WideBus bus_in = in.read(); WideBus bus_out;
            PROCESS_NONLINEAR_BUS: for (int j = 0; j < PARALLEL_FACTOR; ++j) {
                // #pragma HLS UNROLL
                #pragma HLS PIPELINE II=1
                data_t x = bus_in.data[j];
                ap_uint<32> result_fp32;
                if (opcode == OP_SILU) {
                    if (bf16_is_nan(x)) { result_fp32 = bf16_2_fp32(BF16_NAN); }
                    else if (bf16_is_inf(x)) { result_fp32 = (x[15] == 1) ? bf16_2_fp32(BF16_NAN) : bf16_2_fp32(x); }
                    else { result_fp32 = silu_core_robust(x); }
                } else { // OP_GELU
                    ap_uint<32> x_fp32 = bf16_2_fp32(x);
                    ap_uint<32> v1 = fp32fma(x_fp32, FP32_SQRT_1_2_BITS, FP32_ZERO_BITS);
                    ap_uint<32> v2_fp32 = nlu_lookup(fp32_2_bf16(v1), NLU_OP_ERF);
                    ap_uint<32> v3 = fp32_add_propagate_specials(v2_fp32, FP32_ONE_BITS);
                    ap_uint<32> v4 = fp32fma(v3, FP32_HALF_BITS, FP32_ZERO_BITS);
                    result_fp32 = fp32fma(x_fp32, v4, FP32_ZERO_BITS);
                }
                bus_out.data[j] = fp32_2_bf16(result_fp32);
            }
            out.write(bus_out);
        }
    } else {
        DRAIN_NONLINEAR: for (int i = 0; i < total_buses; ++i) {
            #pragma HLS PIPELINE II=1
            in.read(); out.write(WideBus());
        }
    }
}*/
// [NEW] A dedicated pipeline ONLY for SILU
void silu_pipeline(hls::stream<WideBus>& in, hls::stream<WideBus>& out) {
    #pragma HLS INLINE off // Prevent HLS from trying to merge it back
    const int total_buses = (N_DIM * D_DIM) / PARALLEL_FACTOR;
    COMPUTE_SILU: for (int i = 0; i < total_buses; ++i) {
        #pragma HLS PIPELINE II=1
        WideBus bus_in = in.read();
        WideBus bus_out;
        PROCESS_SILU_BUS: for (int j = 0; j < PARALLEL_FACTOR; ++j) {
            #pragma HLS UNROLL
            ap_uint<32> result_fp32 = silu_core_robust(bus_in.data[j]);
            bus_out.data[j] = fp32_2_bf16(result_fp32);
        }
        out.write(bus_out);
    }
}

// [NEW] A dedicated pipeline ONLY for GELU
void gelu_pipeline(hls::stream<WideBus>& in, hls::stream<WideBus>& out) {
    #pragma HLS INLINE off // Prevent HLS from trying to merge it back
    const int total_buses = (N_DIM * D_DIM) / PARALLEL_FACTOR;
    COMPUTE_GELU: for (int i = 0; i < total_buses; ++i) {
        #pragma HLS PIPELINE II=1
        WideBus bus_in = in.read();
        WideBus bus_out;
        PROCESS_GELU_BUS: for (int j = 0; j < PARALLEL_FACTOR; ++j) {
            #pragma HLS UNROLL
            data_t x = bus_in.data[j];
            ap_uint<32> x_fp32 = bf16_2_fp32(x);
            ap_uint<32> v1 = fp32fma(x_fp32, FP32_SQRT_1_2_BITS, FP32_ZERO_BITS);
            ap_uint<32> v2_fp32 = nlu_lookup(fp32_2_bf16(v1), NLU_OP_ERF);
            ap_uint<32> v3 = fp32_add_propagate_specials(v2_fp32, FP32_ONE_BITS);
            ap_uint<32> v4 = fp32fma(v3, FP32_HALF_BITS, FP32_ZERO_BITS);
            ap_uint<32> result_fp32 = fp32fma(x_fp32, v4, FP32_ZERO_BITS);
            bus_out.data[j] = fp32_2_bf16(result_fp32);
        }
        out.write(bus_out);
    }
}

// [MODIFIED] The top-level nonlinear pipeline now acts as a simple dispatcher
void nonlinear_pipeline(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    // This function now simply calls the correct specialized pipeline.
    // This structure prevents HLS from synthesizing hardware for both paths simultaneously.
    if (opcode == OP_SILU) {
        silu_pipeline(in, out);
    } else if (opcode == OP_GELU) {
        gelu_pipeline(in, out);
    } else {
        // Drain loop for cases where this pipeline is not used
        const int total_buses = (N_DIM * D_DIM) / PARALLEL_FACTOR;
        DRAIN_NONLINEAR: for (int i = 0; i < total_buses; ++i) {
            #pragma HLS PIPELINE II=1
            in.read();
            out.write(WideBus());
        }
    }
}
// This new pipeline will handle both LayerNorm and RMSNorm
void unified_norm_pipeline(hls::stream<data_t>& in, hls::stream<data_t>& out, ActivationOpcode opcode) {
    hls::stream<data_t> in_row("in_row_un");
    hls::stream<data_t> out_row("out_row_un");
    #pragma HLS STREAM variable=in_row  depth=D_DIM
    #pragma HLS STREAM variable=out_row depth=D_DIM

    for (int n = 0; n < N_DIM; ++n) {
        #pragma HLS LOOP_FLATTEN off
        for(int i = 0; i < D_DIM; ++i) {
            #pragma HLS PIPELINE
            in_row.write(in.read());
        }
        // The opcode determines which core logic is executed internally
        if (opcode == OP_LAYERNORM) {
            layernorm_row_core(in_row, out_row);
        } else {
            rmsnorm_row_core(in_row, out_row);
        }
        for(int i = 0; i < D_DIM; ++i) {
            #pragma HLS PIPELINE
            out.write(out_row.read());
        }
    }
}
// New wrapped pipeline for the top level
void wrapped_unified_norm_pipeline(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    if (opcode == OP_LAYERNORM || opcode == OP_RMSNORM) {
        unified_norm_dataflow_core(in, out, opcode);
    } else {
        // Drain loop for the wrapper
        for (int i = 0; i < (N_DIM*D_DIM)/PARALLEL_FACTOR; ++i) {
            #pragma HLS PIPELINE II=1
            in.read();
            out.write(WideBus());
        }
    }
}
void BusToStream_Adapter(hls::stream<WideBus>& bus_in, hls::stream<data_t>& serial_out) {
    #pragma HLS INLINE off
    for (int i = 0; i < (N_DIM * D_DIM) / PARALLEL_FACTOR; ++i) {
        WideBus bus = bus_in.read();
        UNPACK_BUS_TO_STREAM: for (int j = 0; j < PARALLEL_FACTOR; ++j) {
            #pragma HLS PIPELINE II=1
            serial_out.write(bus.data[j]);
        }
    }
}
void StreamToBus_Adapter(hls::stream<data_t>& serial_in, hls::stream<WideBus>& bus_out) {
    #pragma HLS INLINE off
    for (int i = 0; i < (N_DIM * D_DIM) / PARALLEL_FACTOR; ++i) {
        WideBus bus;
        PACK_STREAM_TO_BUS: for (int j = 0; j < PARALLEL_FACTOR; ++j) {
            #pragma HLS PIPELINE II=1
            bus.data[j] = serial_in.read();
        }
        bus_out.write(bus);
    }
}
void softmax_dataflow_core(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    #pragma HLS DATAFLOW
    hls::stream<data_t> serial_in("softmax_serial_in");
    hls::stream<data_t> serial_out("softmax_serial_out");
    #pragma HLS STREAM variable=serial_in  depth=D_DIM
    #pragma HLS STREAM variable=serial_out depth=D_DIM
    BusToStream_Adapter(in, serial_in);
    softmax_pipeline(serial_in, serial_out, opcode);
    StreamToBus_Adapter(serial_out, out);
}
void layernorm_dataflow_core(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    #pragma HLS DATAFLOW
    hls::stream<data_t> serial_in("layernorm_serial_in");
    hls::stream<data_t> serial_out("layernorm_serial_out");
    #pragma HLS STREAM variable=serial_in  depth=D_DIM
    #pragma HLS STREAM variable=serial_out depth=D_DIM
    BusToStream_Adapter(in, serial_in);
    layernorm_pipeline(serial_in, serial_out, opcode);
    StreamToBus_Adapter(serial_out, out);
}
void rmsnorm_dataflow_core(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    #pragma HLS DATAFLOW
    hls::stream<data_t> serial_in("rmsnorm_serial_in");
    hls::stream<data_t> serial_out("rmsnorm_serial_out");
    #pragma HLS STREAM variable=serial_in  depth=D_DIM
    #pragma HLS STREAM variable=serial_out depth=D_DIM
    BusToStream_Adapter(in, serial_in);
    rmsnorm_pipeline(serial_in, serial_out, opcode);
    StreamToBus_Adapter(serial_out, out);
}
void wrapped_softmax_pipeline(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    if (opcode == OP_SOFTMAX) {
        softmax_dataflow_core(in, out, opcode);
    } else {
        for (int i = 0; i < (N_DIM*D_DIM)/PARALLEL_FACTOR; ++i) {
            #pragma HLS PIPELINE II=1
            in.read(); out.write(WideBus());
        }
    }
}
void wrapped_layernorm_pipeline(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    if (opcode == OP_LAYERNORM) {
        layernorm_dataflow_core(in, out, opcode);
    } else {
        for (int i = 0; i < (N_DIM*D_DIM)/PARALLEL_FACTOR; ++i) {
            #pragma HLS PIPELINE II=1
            in.read(); out.write(WideBus());
        }
    }
}
void wrapped_rmsnorm_pipeline(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    if (opcode == OP_RMSNORM) {
        rmsnorm_dataflow_core(in, out, opcode);
    } else {
        for (int i = 0; i < (N_DIM*D_DIM)/PARALLEL_FACTOR; ++i) {
            #pragma HLS PIPELINE II=1
            in.read(); out.write(WideBus());
        }
    }
}
// New dataflow core for the unified pipeline
void unified_norm_dataflow_core(hls::stream<WideBus>& in, hls::stream<WideBus>& out, ActivationOpcode opcode) {
    #pragma HLS DATAFLOW
    hls::stream<data_t> serial_in("unified_norm_serial_in");
    hls::stream<data_t> serial_out("unified_norm_serial_out");
    #pragma HLS STREAM variable=serial_in  depth=D_DIM
    #pragma HLS STREAM variable=serial_out depth=D_DIM

    BusToStream_Adapter(in, serial_in);
    unified_norm_pipeline(serial_in, serial_out, opcode);
    StreamToBus_Adapter(serial_out, out);
}
// ===================================================================================
//  SECTION 5: TOP-LEVEL SCHEDULER (MODIFIED for Resource Reuse)
// ===================================================================================

/**
 * @brief Routes top-level inputs to the four parallel pipelines.
 * @note MODIFIED: lay_in and rms_in are replaced by a single uni_in.
 */
/*void accelerator_router(
    hls::stream<WideBus>& top_in_a, hls::stream<WideBus>& top_in_b,
    hls::stream<WideBus>& ele_in_a, hls::stream<WideBus>& ele_in_b,
    hls::stream<WideBus>& nli_in,   hls::stream<WideBus>& smx_in,
    hls::stream<WideBus>& uni_in
) {
    // "Dumb" router: Unconditionally broadcast inputs to all pipelines.
    for (int i = 0; i < (N_DIM * D_DIM) / PARALLEL_FACTOR; ++i) {
        #pragma HLS PIPELINE II=1
        WideBus val_a = top_in_a.read();
        WideBus val_b = top_in_b.read();
        
        ele_in_a.write(val_a);
        ele_in_b.write(val_b);
        nli_in.write(val_a);
        smx_in.write(val_a);
        uni_in.write(val_a); // Route to the unified norm pipeline
    }
}*/

/**
 * @brief Multiplexes results from the four parallel pipelines.
 * @note MODIFIED: lay_out and rms_out are replaced by a single uni_out.
 */
/*void accelerator_mux(
    hls::stream<WideBus>& ele_out, hls::stream<WideBus>& nli_out,
    hls::stream<WideBus>& smx_out, hls::stream<WideBus>& uni_out,
    hls::stream<WideBus>& top_out, ActivationOpcode opcode
) {
    // "Dumb" Mux: Unconditionally read from all pipelines.
    for (int i = 0; i < (N_DIM * D_DIM) / PARALLEL_FACTOR; ++i) {
        #pragma HLS PIPELINE II=1
        // Read all potential results
        WideBus ele_res = ele_out.read();
        WideBus nli_res = nli_out.read();
        WideBus smx_res = smx_out.read();
        WideBus uni_res = uni_out.read(); // Read from the unified norm pipeline
        
        // Use a switch to select the correct result to forward.
        WideBus final_res;
        switch(opcode) {
            case OP_ADD: case OP_MUL:       final_res = ele_res; break;
            case OP_SILU: case OP_GELU:      final_res = nli_res; break;
            case OP_SOFTMAX:                 final_res = smx_res; break;
            // MODIFIED: Both LayerNorm and RMSNorm now source their result from uni_res
            case OP_LAYERNORM:               final_res = uni_res; break;
            case OP_RMSNORM:                 final_res = uni_res; break;
            default:                         final_res = ele_res; break; // Default case
        }
        top_out.write(final_res);
    }
}*/

/**
 * @brief Top-level dataflow region orchestrating the parallel execution of activation functions.
 * @note  MODIFIED: Implements a 4-pipeline structure (Elementwise, Nonlinear, Softmax, UnifiedNorm)
 *        to reuse hardware resources between LayerNorm and RMSNorm.
 */
/*void accelerator_core_logic(
    hls::stream<WideBus>& stream_in_a, hls::stream<WideBus>& stream_in_b,
    hls::stream<WideBus>& stream_out, ActivationOpcode opcode
) {
    #pragma HLS DATAFLOW

    // --- Internal Streams Declaration ---
    // Streams for the three independent pipelines
    hls::stream<WideBus> ele_in_a, ele_in_b, ele_out;
    hls::stream<WideBus> nli_in, nli_out;
    hls::stream<WideBus> smx_in, smx_out;
    // NEW: Streams for the single, unified Normalization pipeline
    hls::stream<WideBus> uni_in, uni_out;

    // --- Stream Depth Configuration ---
    const int BUS_STREAM_DEPTH = 256; 
    #pragma HLS STREAM variable=ele_in_a depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=ele_in_b depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=nli_in   depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=smx_in   depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=uni_in   depth=BUS_STREAM_DEPTH // NEW
    #pragma HLS STREAM variable=ele_out  depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=nli_out  depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=smx_out  depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=uni_out  depth=BUS_STREAM_DEPTH // NEW

    // 1. Unconditionally route inputs to all four parallel pipelines
    accelerator_router(stream_in_a, stream_in_b, ele_in_a, ele_in_b, nli_in, smx_in, uni_in);

    // 2. Instantiate all "smart" pipelines. They will run concurrently.
    elementwise_pipeline(ele_in_a, ele_in_b, ele_out, opcode);
    nonlinear_pipeline(nli_in, nli_out, opcode);
    wrapped_softmax_pipeline(smx_in, smx_out, opcode);
    // NEW: A single wrapped pipeline now handles both LayerNorm and RMSNorm.
    wrapped_unified_norm_pipeline(uni_in, uni_out, opcode);

    // 3. Mux the results from the four pipelines.
    accelerator_mux(ele_out, nli_out, smx_out, uni_out, stream_out, opcode);
}*/
/**
 * @brief Top-level compute logic, re-architected as a top-level dispatcher.
 * @note  This new design ABANDONS the parallel dataflow model (`#pragma HLS DATAFLOW`) 
 *        in favor of a sequential, mutually exclusive execution model. It uses a 
 *        'switch' statement to explicitly call ONLY the required pipeline for a 
 *        given opcode. This robustly prevents the dataflow deadlocks that occurred 
 *        when fast and slow pipelines ran concurrently.
 */
void accelerator_core_logic(
    hls::stream<WideBus>& stream_in_a, 
    hls::stream<WideBus>& stream_in_b,
    hls::stream<WideBus>& stream_out, 
    ActivationOpcode opcode
) {
    // This top-level function is NO LONGER a DATAFLOW region.
    // It acts as a controller that sequentially invokes the correct pipeline.
    
    // We instantiate placeholder streams. The compiler will optimize away
    // any streams that are not used in a given 'switch' branch.
    hls::stream<WideBus> ele_out, nli_out, smx_out, uni_out;
    const int BUS_STREAM_DEPTH = 2; // Depth can be minimal as there's no concurrency
    #pragma HLS STREAM variable=ele_out depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=nli_out depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=smx_out depth=BUS_STREAM_DEPTH
    #pragma HLS STREAM variable=uni_out depth=BUS_STREAM_DEPTH

    switch(opcode) {
        case OP_ADD:
        case OP_MUL: {
            // For ADD/MUL, only the elementwise_pipeline is called and connected.
            elementwise_pipeline(stream_in_a, stream_in_b, stream_out, opcode);
            break;
        }
        case OP_SILU:
        case OP_GELU: {
            // For SILU/GELU, only the nonlinear_pipeline is called.
            // Note: stream_in_b is unused but must be drained to prevent upstream stalls.
            nonlinear_pipeline(stream_in_a, stream_out, opcode);
            
            // Draining the unused input stream is critical!
            const int total_buses = (N_DIM * D_DIM) / PARALLEL_FACTOR;
            DRAIN_B_IN_NLI: for (int i = 0; i < total_buses; ++i) {
                #pragma HLS PIPELINE II=1
                stream_in_b.read();
            }
            break;
        }
        case OP_SOFTMAX: {
            // For Softmax, only its wrapped pipeline is called.
            wrapped_softmax_pipeline(stream_in_a, stream_out, opcode);

            // Drain the unused input stream
            const int total_buses = (N_DIM * D_DIM) / PARALLEL_FACTOR;
            DRAIN_B_IN_SMX: for (int i = 0; i < total_buses; ++i) {
                #pragma HLS PIPELINE II=1
                stream_in_b.read();
            }
            break;
        }
        case OP_LAYERNORM:
        case OP_RMSNORM: {
            // For Norms, only the unified norm pipeline is called.
            wrapped_unified_norm_pipeline(stream_in_a, stream_out, opcode);

            // Drain the unused input stream
            const int total_buses = (N_DIM * D_DIM) / PARALLEL_FACTOR;
            DRAIN_B_IN_UNI: for (int i = 0; i < total_buses; ++i) {
                #pragma HLS PIPELINE II=1
                stream_in_b.read();
            }
            break;
        }
        default: {
            // Default case: Drain both inputs and produce no output.
            // This prevents deadlocks if an invalid opcode is provided.
            const int total_buses = (N_DIM * D_DIM) / PARALLEL_FACTOR;
            DRAIN_DEFAULT: for (int i = 0; i < total_buses; ++i) {
                #pragma HLS PIPELINE II=1
                stream_in_a.read();
                stream_in_b.read();
                // You might need to write dummy data if the downstream expects it,
                // but in this architecture, the connection is direct.
            }
            break;
        }
    }
}
// ===================================================================================
//  NEW: Adapter functions for 128-bit Interface to 64-bit Processing bus
// ===================================================================================

/**
 * @brief BRAM-to-Stream with Unpack: Reads 128-bit InterfaceBus from BRAM,
 *        unpacks it into two 64-bit WideBus objects, and writes them to a stream.
 */
static void bram_to_stream_unpack(
    InterfaceBus* bram_in,
    hls::stream<WideBus>& stream_out,
    int total_interface_buses
) {
BRAM2STREAM_UNPACK_LOOP:
    for (int i = 0; i < total_interface_buses; ++i) {
        #pragma HLS PIPELINE II=2 // This pipeline achieves II=1 for the output stream
        InterfaceBus temp = bram_in[i];
        
        // Unpack the 128-bit bus into two 64-bit buses
        WideBus p_bus_1, p_bus_2;
        UNPACK_SUB_LOOP_1: for (int k = 0; k < PARALLEL_FACTOR; ++k) {
            #pragma HLS UNROLL
            p_bus_1.data[k] = temp.data[k];
        }
        UNPACK_SUB_LOOP_2: for (int k = 0; k < PARALLEL_FACTOR; ++k) {
            #pragma HLS UNROLL
            p_bus_2.data[k] = temp.data[PARALLEL_FACTOR + k];
        }

        stream_out.write(p_bus_1);
        stream_out.write(p_bus_2);
    }
}

/**
 * @brief Stream-to-BRAM with Pack: Reads two 64-bit WideBus objects from a stream,
 *        packs them into one 128-bit InterfaceBus, and writes it to BRAM.
 */
static void stream_to_bram_pack(
    hls::stream<WideBus>& stream_in,
    InterfaceBus* bram_out,
    int total_interface_buses
) {
STREAM2BRAM_PACK_LOOP:
    for (int i = 0; i < total_interface_buses; ++i) {
        #pragma HLS PIPELINE II=2 // This pipeline reads two items from the stream
        
        WideBus p_bus_1 = stream_in.read();
        WideBus p_bus_2 = stream_in.read();

        InterfaceBus temp;
        // Pack two 64-bit buses into one 128-bit bus
        PACK_SUB_LOOP_1: for (int k = 0; k < PARALLEL_FACTOR; ++k) {
            #pragma HLS UNROLL
            temp.data[k] = p_bus_1.data[k];
        }
        PACK_SUB_LOOP_2: for (int k = 0; k < PARALLEL_FACTOR; ++k) {
            #pragma HLS UNROLL
            temp.data[PARALLEL_FACTOR + k] = p_bus_2.data[k];
        }
        
        bram_out[i] = temp;
    }
}
// [NEW] Direct, 1-to-1 adapter from BRAM to Stream for 8-wide parallelism
static void bram_to_stream_direct(
    InterfaceBus* bram_in,
    hls::stream<WideBus>& stream_out,
    int total_interface_buses
) {
BRAM2STREAM_DIRECT_LOOP:
    for (int i = 0; i < total_interface_buses; ++i) {
        #pragma HLS PIPELINE II=1
        InterfaceBus temp = bram_in[i];
        WideBus bus_out;
        // Direct copy since widths match
        for (int k = 0; k < INTERFACE_FACTOR; ++k) {
            #pragma HLS UNROLL
            bus_out.data[k] = temp.data[k];
        }
        stream_out.write(bus_out);
    }
}

// [NEW] Direct, 1-to-1 adapter from Stream to BRAM for 8-wide parallelism
static void stream_to_bram_direct(
    hls::stream<WideBus>& stream_in,
    InterfaceBus* bram_out,
    int total_interface_buses
) {
STREAM2BRAM_DIRECT_LOOP:
    for (int i = 0; i < total_interface_buses; ++i) {
        #pragma HLS PIPELINE II=1
        WideBus bus_in = stream_in.read();
        InterfaceBus temp;
        // Direct copy since widths match
        for (int k = 0; k < INTERFACE_FACTOR; ++k) {
            #pragma HLS UNROLL
            temp.data[k] = bus_in.data[k];
        }
        bram_out[i] = temp;
    }
}
/**
 * @brief Pass-through logic for Co-simulation fast mode, adapted for the new bus types.
 */
static void fast_mode_pass_through(
    hls::stream<WideBus>& stream_a_in,
    hls::stream<WideBus>& stream_b_in,
    hls::stream<WideBus>& stream_out,
    int total_processing_buses
) {
PASS_THROUGH_FAST_MODE:
    for (int i = 0; i < total_processing_buses; ++i) {
        #pragma HLS PIPELINE II=1
        stream_a_in.read();
        stream_b_in.read();
        stream_out.write(WideBus()); // Write an empty bus
    }
}

static void compute_stage_dataflow(
    InterfaceBus bram_in_a[],
    InterfaceBus bram_in_b[],
    InterfaceBus bram_out[],
    ActivationOpcode opcode,
    int total_interface_buses
) {
    #pragma HLS DATAFLOW

    // The total number of *processing buses* is now equal to the total *interface buses*
    const int total_processing_buses = total_interface_buses;

    hls::stream<WideBus> internal_stream_a("internal_stream_a");
    hls::stream<WideBus> internal_stream_b("internal_stream_b");
    hls::stream<WideBus> internal_stream_out("internal_stream_out");
    const int STREAM_DEPTH = 32;
    #pragma HLS STREAM variable=internal_stream_a depth=STREAM_DEPTH
    #pragma HLS STREAM variable=internal_stream_b depth=STREAM_DEPTH
    #pragma HLS STREAM variable=internal_stream_out depth=STREAM_DEPTH

    // 1. Call the new direct BRAM-to-Stream adapters
    bram_to_stream_direct(bram_in_a, internal_stream_a, total_interface_buses);
    bram_to_stream_direct(bram_in_b, internal_stream_b, total_interface_buses);

    // 2. Core Compute Logic (Unchanged)
    accelerator_core_logic(
        internal_stream_a,
        internal_stream_b,
        internal_stream_out,
        opcode
    );

    // 3. Call the new direct Stream-to-BRAM adapter
    stream_to_bram_direct(internal_stream_out, bram_out, total_interface_buses);
}
// ===================================================================================
//  FINAL TOP-LEVEL FUNCTION with URAM for Interface Buffers
// ===================================================================================
void activation_accelerator(
    InterfaceBus* in0,
    InterfaceBus* in1,
    InterfaceBus* out,
    int stage,   // 在左，对应低地址 0x34 (PYNQ)
    int config   // 在右，对应高地址 0x3C (PYNQ)
) {
    const int total_interface_buses = (N_DIM * D_DIM) / INTERFACE_FACTOR;

    // AXI Master 接口
    #pragma HLS INTERFACE m_axi port=in0   offset=slave bundle=gmem0 depth=total_interface_buses
    #pragma HLS INTERFACE m_axi port=in1   offset=slave bundle=gmem1 depth=total_interface_buses
    #pragma HLS INTERFACE m_axi port=out   offset=slave bundle=gmem2 depth=total_interface_buses

    // AXI-Lite Slave 接口
    #pragma HLS INTERFACE s_axilite port=in0 bundle=control
    #pragma HLS INTERFACE s_axilite port=in1 bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=config bundle=control
    #pragma HLS INTERFACE s_axilite port=stage bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // 内部 BRAM/URAM 缓冲区，命名与端口保持一致
    static InterfaceBus bram_in0[total_interface_buses];
    #pragma HLS BIND_STORAGE variable=bram_in0 type=RAM_2P impl=URAM

    static InterfaceBus bram_in1[total_interface_buses];
    #pragma HLS BIND_STORAGE variable=bram_in1 type=RAM_2P impl=URAM
    
    static InterfaceBus bram_out[total_interface_buses];
    #pragma HLS BIND_STORAGE variable=bram_out type=RAM_2P impl=URAM

    // 关键性能指令，确保变量名与上面声明的完全一致
    #pragma HLS dependence variable=bram_in0 inter false
    #pragma HLS dependence variable=bram_in1 inter false
    #pragma HLS dependence variable=bram_out inter false
    
    switch (stage) {
        case 0: { // Stage 0: 从 DDR 加载到片上 URAM
        LOAD_IN0_LOOP:
            for (int i = 0; i < total_interface_buses; ++i) {
                #pragma HLS PIPELINE II=1
                bram_in0[i] = in0[i]; // 使用 bram_in0
            }
        LOAD_IN1_LOOP:
            for (int i = 0; i < total_interface_buses; ++i) {
                #pragma HLS PIPELINE II=1
                bram_in1[i] = in1[i]; // 使用 bram_in1
            }
            break;
        }

        case 1: { // Stage 1: 调用计算核心
            compute_stage_dataflow(
                bram_in0, // 传递 bram_in0
                bram_in1, // 传递 bram_in1
                bram_out, 
                (ActivationOpcode)config, 
                total_interface_buses
            );
            break;
        }

        case 2: { // Stage 2: 从片上 URAM 存储到 DDR
        STORE_LOOP:
            for (int i = 0; i < total_interface_buses; ++i) {
                #pragma HLS PIPELINE II=1
                out[i] = bram_out[i];
            }
            break;
        }

        default: {
            break;
        }
    }
}
// in accelerator.cpp
