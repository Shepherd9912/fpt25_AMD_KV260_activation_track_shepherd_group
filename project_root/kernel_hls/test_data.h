#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <cstdint>

extern const uint16_t hls_input_X[49152];
extern const uint16_t hls_input_Y[49152];
extern const uint16_t hls_golden_add[49152];
extern const uint16_t hls_golden_mul[49152];
extern const uint16_t hls_golden_gelu[49152];
extern const uint16_t hls_golden_silu[49152];
extern const uint16_t hls_golden_softmax[49152];
extern const uint16_t hls_golden_layernorm[49152];
extern const uint16_t hls_golden_rmsnorm[49152];

#endif
