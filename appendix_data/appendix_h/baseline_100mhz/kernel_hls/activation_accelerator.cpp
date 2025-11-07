#include "activation_accelerator.h"
#include <iostream>
#include <cmath>
#include <hls_math.h>

/*
0 eltwise add
1 safe softmax
2 mask safe softmax
3 sigmoid
4 silu
5 rms norm
6 layer norm
7 eltwise mul
*/

// bf16 helper and conversion functions
// (No changes from previous correct version, adding pragmas for performance)

void bf16_to_float(const uint16* in, float* out, int len) {
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        uint32_t x_f32 = ((uint32_t)in[i]) << 16;
        out[i] = *(float*)&x_f32;
    }
}

void float_to_bf16(const float* in, uint16* out, int len){
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        const float val = in[i];
        uint32_t* y_f32_ptr = (uint32_t*)&val;
        out[i] = (*y_f32_ptr) >> 16;
    }
}

void float_sigmoid(const float* x, float* y, int len) {
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        y[i] = 1.0f / (1.0f + hls::expf(-x[i]));
    }
}

void float_silu(const float* x, float* y, int len) {
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        y[i] = x[i] * (1.0f / (1.0f + hls::expf(-x[i])));
    }
}

void float_rms_norm(const float* x, float* y, int len) {
    const float eps = 1e-5f; // Match Python
    float sum_sq = 0.0f;
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        sum_sq += x[i] * x[i];
    }
    float rms = hls::sqrtf(sum_sq / len + eps);
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        y[i] = x[i] / rms;
    }
}

void float_layer_norm(const float* x, float* y, int len) {
    const float eps = 1e-5f; // Match Python
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        sum += x[i];
    }
    float mean = sum / len;
    float var = 0.0f;
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        float diff = x[i] - mean;
        var += diff * diff;
    }
    float stddev = hls::sqrtf(var / len + eps);
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        y[i] = (x[i] - mean) / stddev;
    }
}

void float_add(const float* x, const float* y, float* out, int len) {
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = x[i] + y[i];
    }
}

void float_mul(const float* x, const float* y, float* out, int len) {
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = x[i] * y[i];
    }
}

void float_safe_softmax(const float* x, float* y, int len) {
    float max_val = -__FLT_MAX__;
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        float exp_val = hls::expf(x[i] - max_val);
        y[i] = exp_val; // Store intermediate result
        sum += exp_val;
    }
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        y[i] = y[i] / sum;
    }
}

void float_mask_safe_softmax(const float* x, const float* mask, float* y, int len) {
    float max_val = -__FLT_MAX__;
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        float masked_val = x[i] + mask[i]; // PyTorch adds -inf for masking
        y[i] = masked_val; // Use y as temporary storage
        if (masked_val > max_val) max_val = masked_val;
    }
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        float exp_val = hls::expf(y[i] - max_val);
        y[i] = exp_val;
        sum += exp_val;
    }
    for (int i = 0; i < len; ++i) {
#pragma HLS PIPELINE II=1
        y[i] = y[i] / sum;
    }
}


void activation_accelerator(uint16* in0, uint16* in1, uint16* out, int32 stage, int32 config) {
#pragma HLS INTERFACE m_axi port=in0 offset=slave bundle=gmem0 depth=49152
#pragma HLS INTERFACE m_axi port=in1 offset=slave bundle=gmem1 depth=49152
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem2 depth=49152
#pragma HLS INTERFACE s_axilite port=stage
#pragma HLS INTERFACE s_axilite port=config
#pragma HLS INTERFACE s_axilite port=return

    static uint16 buf0[DATA_SIZE];
    #pragma HLS BIND_STORAGE variable=buf0 type=RAM_T2P impl=URAM
    static uint16 buf1[DATA_SIZE];
    #pragma HLS BIND_STORAGE variable=buf1 type=RAM_T2P impl=URAM
    static uint16 buf2[DATA_SIZE];
    #pragma HLS BIND_STORAGE variable=buf2 type=RAM_T2P impl=URAM
    
    // Use float for all intermediate calculations for better precision
    static float x_float[DATA_SIZE], y_float[DATA_SIZE], out_float[DATA_SIZE];

    if(stage == STAGE_LOAD) {
        for(int i = 0; i < DATA_SIZE; i++) {
#pragma HLS PIPELINE II=1
            buf0[i] = in0[i];
            buf1[i] = in1[i];
        }
    }
    
    if(stage == STAGE_COMPUTE) {
        bf16_to_float(buf0, x_float, DATA_SIZE);
        bf16_to_float(buf1, y_float, DATA_SIZE);

        if(config == CONFIG_ELTWISE_ADD)        float_add(x_float, y_float, out_float, DATA_SIZE);
        else if(config == CONFIG_SAFE_SOFTMAX)  float_safe_softmax(x_float, out_float, DATA_SIZE);
        else if(config == CONFIG_MASK_SOFTMAX)  float_mask_safe_softmax(x_float, y_float, out_float, DATA_SIZE);
        else if(config == CONFIG_SIGMOID)       float_sigmoid(x_float, out_float, DATA_SIZE);
        else if(config == CONFIG_SILU)          float_silu(x_float, out_float, DATA_SIZE);
        else if(config == CONFIG_RMS_NORM)      float_rms_norm(x_float, out_float, DATA_SIZE);
        else if(config == CONFIG_LAYER_NORM)    float_layer_norm(x_float, out_float, DATA_SIZE);
        else if(config == CONFIG_ELTWISE_MUL)   float_mul(x_float, y_float, out_float, DATA_SIZE);
        
        float_to_bf16(out_float, buf2, DATA_SIZE);
    }
    
    if(stage == STAGE_STORE) {
        for(int i = 0; i < DATA_SIZE; i++) {
#pragma HLS PIPELINE II=1
            out[i] = buf2[i];
        }
    }
}