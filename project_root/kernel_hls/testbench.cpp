#include "accelerator.h" // Includes our DUT, WideBus, PARALLEL_FACTOR, etc.
#include "test_data.h"    // Includes the golden data arrays

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <limits>
#include <set>
#include <cstdio>

// =================================================================================
// 1. Constants and Data Buffers (UNCHANGED)
// =================================================================================

const int TENSOR_SIZE = N_DIM * D_DIM;

// Memory buffers to hold our test data. These now simulate DDR memory.
uint16_t hls_in_a[TENSOR_SIZE];
uint16_t hls_in_b[TENSOR_SIZE];
uint16_t hls_out[TENSOR_SIZE];

// =================================================================================
// 2. Accuracy Verification Functions (UNCHANGED)
// =================================================================================

// Helper to convert bf16 bits to a standard float for comparison
float bf16_bits_to_float(uint16_t bits) {
    uint32_t bits_32 = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &bits_32, sizeof(float));
    return result;
}

// This function remains IDENTICAL. It works on flat arrays, which is perfect.
float calculate_l2_error(const uint16_t* hls_result, const uint16_t* golden_ref, int size) {
    // ... (Your original, excellent implementation here, no changes needed) ...
    double sum_diff_sq = 0.0;
    double sum_golden_sq = 0.0;
    bool fatal_mismatch_found = false;

    for (int i = 0; i < size; ++i) {
        float hls_float = bf16_bits_to_float(hls_result[i]);
        float golden_float = bf16_bits_to_float(golden_ref[i]);

        bool is_golden_nan = std::isnan(golden_float);
        bool is_hls_nan = std::isnan(hls_float);

        if (is_golden_nan) {
            if (!is_hls_nan) {
                if (!fatal_mismatch_found) {
                    std::cout << "  ERROR: Special value mismatch at index " << i << ". "
                              << "Expected: nan, Got: " << hls_float << std::endl;
                    fatal_mismatch_found = true;
                }
                sum_diff_sq += 1e20;
            }
        } else if (std::isinf(golden_float)) {
            if (hls_float != golden_float) {
                if (!fatal_mismatch_found) {
                    std::cout << "  ERROR: Special value mismatch at index " << i << ". "
                              << "Expected: " << golden_float << ", Got: " << hls_float << std::endl;
                    fatal_mismatch_found = true;
                }
                sum_diff_sq += 1e20;
            }
        } else {
            if (is_hls_nan || std::isinf(hls_float)) {
                 if (!fatal_mismatch_found) {
                    std::cout << "  ERROR: Special value mismatch at index " << i << ". "
                              << "Expected: " << golden_float << ", Got: " << hls_float << std::endl;
                    fatal_mismatch_found = true;
                }
                sum_diff_sq += 1e20;
            } else {
                double diff = static_cast<double>(hls_float) - static_cast<double>(golden_float);
                sum_diff_sq += diff * diff;
            }
            sum_golden_sq += static_cast<double>(golden_float) * static_cast<double>(golden_float);
        }
    }

    if (fatal_mismatch_found) {
        return std::numeric_limits<float>::infinity();
    }

    if (sum_golden_sq < 1e-20) {
        return (sum_diff_sq < 1e-20) ? 0.0f : static_cast<float>(std::sqrt(sum_diff_sq));
    }

    return static_cast<float>(std::sqrt(sum_diff_sq) / std::sqrt(sum_golden_sq));
}


// =================================================================================
// 3. Main Test Driver (MODIFIED FOR AXI-MM POINTER INTERFACE)
// =================================================================================

int main() {
    std::cout << "--- Starting HLS Testbench for AXI-MM Pointer-Based Accelerator ---" << std::endl;

    struct TestConfig {
        ActivationOpcode config_id;
        std::string name;
        const uint16_t* golden_data;
    };

    std::vector<TestConfig> tests = {
        {OP_ADD,       "Elementwise Add", hls_golden_add},
        {OP_MUL,       "Elementwise Mul", hls_golden_mul},
        {OP_SILU,      "SiLU",            hls_golden_silu},
        {OP_GELU,      "GELU",            hls_golden_gelu},
        {OP_RMSNORM,   "RMSNorm",         hls_golden_rmsnorm},
        {OP_LAYERNORM, "LayerNorm",       hls_golden_layernorm},
        {OP_SOFTMAX,   "Softmax",         hls_golden_softmax}
    };

    int failed_tests = 0;
    const float ACCURACY_THRESHOLD = 1e-2;
    const float ROW_ERROR_THRESHOLD = 1e-1;

    // Load test data into our host-side buffers (simulating DDR)
    std::memcpy(hls_in_a, hls_input_X, TENSOR_SIZE * sizeof(uint16_t));
    std::memcpy(hls_in_b, hls_input_Y, TENSOR_SIZE * sizeof(uint16_t));

    // ==========================================================
    // MODIFICATION 1: From Streams to Pointers
    // We now create pointers to our data buffers. `reinterpret_cast` is used
    // to safely convert our uint16_t array pointers to the WideBus pointers
    // that the DUT expects. This is valid because WideBus is just an
    // array of uint16_t, so the memory layout is compatible.
    // ==========================================================
    InterfaceBus* wb_in_a = reinterpret_cast<InterfaceBus*>(hls_in_a);
    InterfaceBus* wb_in_b = reinterpret_cast<InterfaceBus*>(hls_in_b);
    InterfaceBus* wb_out  = reinterpret_cast<InterfaceBus*>(hls_out);
    
    // --- Stage 1: Run the DUT (This code is common to both modes) ---
    std::cout << "--- Running DUT: Stage 0 (Load) ---" << std::endl;
    activation_accelerator(wb_in_a, wb_in_b, wb_out, 0, 0); 


    for (const auto& test : tests) {
        std::cout << "\n=====================================================" << std::endl;
        std::cout << "  TESTING OPCODE: " << test.config_id << " (" << test.name << ")" << std::endl;
        std::cout << "=====================================================" << std::endl;

        // Clear the output buffer before each run to ensure results are from the current test
        std::memset(hls_out, 0, TENSOR_SIZE * sizeof(uint16_t));

        // ==========================================================
        // MODIFICATION 2: Removed Stream Packing Loop
        // The data is already in our `hls_in_a` and `hls_in_b` arrays.
        // We don't need to manually pack it into streams anymore.
        // ==========================================================

        
        
        std::cout << "--- Running DUT: Stage 1 (Compute) ---" << std::endl;
        activation_accelerator(wb_in_a, wb_in_b, wb_out, 1, test.config_id);
        
        std::cout << "--- Running DUT: Stage 2 (Store) ---" << std::endl;
        activation_accelerator(wb_in_a, wb_in_b, wb_out, 2, test.config_id);

        // ==========================================================
        // MODIFICATION 4: Removed Stream Unpacking Loop
        // The DUT writes its results directly into the `hls_out` buffer via
        // the `wb_out` pointer. When the function returns, `hls_out` is
        // already populated. No unpacking is needed.
        // ==========================================================

        std::cout << "--- Comparing Results ---" << std::endl;
        float error = calculate_l2_error(hls_out, test.golden_data, TENSOR_SIZE);

        bool pass = error < ACCURACY_THRESHOLD;

        if (pass) {
            std::cout << "--- Opcode " << test.config_id << " PASSED (L2 Error = " << error << ") ---" << std::endl;
        } else {
            std::cout << "--- ERROR found in Test for Opcode " << test.config_id << " ---" << std::endl;
            std::cout << "  FAIL: L2 Error = " << error << " > " << ACCURACY_THRESHOLD << std::endl;
            failed_tests++;
            
            // NOTE: The detailed reporting logic below does NOT need to change.
            // It correctly operates on the `hls_out` array.
            std::cout << "\n  !!! Detailed Mismatch Report for " << test.name << " !!!" << std::endl;
            
            const int MAX_DIFFERENT_ROWS_TO_PRINT = 30;
            const int ELEMENTS_PER_ROW_TO_PRINT = 6;
            
            std::set<int> reported_rows;

            for (int row = 0; row < N_DIM; ++row) {
                float row_error = calculate_l2_error(&hls_out[row * D_DIM], &test.golden_data[row * D_DIM], D_DIM);
                if (row_error > ROW_ERROR_THRESHOLD) {
                    reported_rows.insert(row);
                }
            }

            if (reported_rows.empty()) {
                 printf("  Overall L2 error is high, but no single row's L2 error exceeded the reporting threshold of %.e\n", ROW_ERROR_THRESHOLD);
            } else {
                int rows_printed_count = 0;
                for (int current_row : reported_rows) {
                    if (rows_printed_count >= MAX_DIFFERENT_ROWS_TO_PRINT) {
                        printf("  ... More failing rows exist but are not shown.\n");
                        break; 
                    }

                    printf("\n  --- Mismatch details for row %d (Row L2 Error > %.e) ---\n", current_row, ROW_ERROR_THRESHOLD);
                    int row_start_idx = current_row * D_DIM;
                    for (int j = row_start_idx; j < row_start_idx + ELEMENTS_PER_ROW_TO_PRINT && j < TENSOR_SIZE; ++j) {
                        bool is_mismatch = (hls_out[j] != test.golden_data[j]);
                        printf("    [%s] Idx %-5d (col %-3d): In_A=0x%04x (%-9.4f), HLS=0x%04x (%-9.4f), Golden=0x%04x (%-9.4f)\n",
                               is_mismatch ? "FAIL" : "pass",
                               j, j % D_DIM,
                               hls_in_a[j], bf16_bits_to_float(hls_in_a[j]),
                               hls_out[j], bf16_bits_to_float(hls_out[j]),
                               test.golden_data[j], bf16_bits_to_float(test.golden_data[j]));
                    }
                    rows_printed_count++;
                }
            }
        }
    }

    std::cout << "\n=====================================================" << std::endl;
    if (failed_tests == 0) {
        std::cout << "All " << tests.size() << " tests PASSED!" << std::endl;
        return 0; // Return 0 for success
    } else {
        std::cout << failed_tests << " out of " << tests.size() << " tests FAILED." << std::endl;
        return 1; // Return 1 for failure
    }
}
