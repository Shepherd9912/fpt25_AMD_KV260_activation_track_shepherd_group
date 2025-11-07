#include "activation_accelerator.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>
#include <unistd.h>
#include <string>
#include <vector> // <--- 关键修正：添加此头文件以使用 std::vector

// 辅助函数：将bf16位模式数组转换为float数组
void bf16_bits_to_float_vector(const uint16* in, float* out, int len) {
    for (int i = 0; i < len; ++i) {
        uint32_t temp = ((uint32_t)in[i]) << 16;
        out[i] = *(float*)&temp;
    }
}

// 从二进制文件加载数据
bool load_binary_data(const std::string& filename, uint16* data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(data), DATA_SIZE * sizeof(uint16_t));
    file.close();
    return true;
}

// 将数据保存到二进制文件
bool save_binary_data(const std::string& filename, uint16* data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot create file for writing: " << filename << std::endl;
        return false;
    }
    file.write(reinterpret_cast<char*>(data), DATA_SIZE * sizeof(uint16_t));
    file.close();
    return true;
}

// 核心修正：引入基于容限的浮点数比较函数
int compare_results_with_tolerance(uint16* hls_result_bits, uint16* golden_bits) {
    // bfloat16的精度较低，因此容差可以适当放宽
    const float relative_tolerance = 1e-2f; // 1% 相对误差
    const float absolute_tolerance = 1e-3f; // 绝对误差

    int errors = 0;
    float max_diff = 0.0f;

    // 创建std::vector来存储转换后的浮点数
    std::vector<float> hls_float(DATA_SIZE);
    std::vector<float> golden_float(DATA_SIZE);
    
    bf16_bits_to_float_vector(hls_result_bits, hls_float.data(), DATA_SIZE);
    bf16_bits_to_float_vector(golden_bits, golden_float.data(), DATA_SIZE);

    for (int i = 0; i < DATA_SIZE; ++i) {
        float hls_val = hls_float[i];
        float golden_val = golden_float[i];
        float diff = std::abs(hls_val - golden_val);

        // 特殊值处理：如果两者都是NaN或都是符号相同的Inf，则认为匹配
        if (std::isnan(hls_val) && std::isnan(golden_val)) {
            continue;
        }
        if (std::isinf(hls_val) && std::isinf(golden_val) && ( (hls_val > 0 && golden_val > 0) || (hls_val < 0 && golden_val < 0) )) {
            continue;
        }
        
        // 核心比较逻辑：如果差异同时大于绝对容差和相对容差，则判为错误
        bool is_error = (diff > absolute_tolerance) && (diff > std::abs(golden_val) * relative_tolerance);

        if (is_error) {
            errors++;
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }

    std::cout << "Max Numerical Difference: " << max_diff << std::endl;
    return errors;
}

// 获取数据路径
std::string get_data_path() {
    std::string rel_path = "../../../../data/"; // 适配Vitis HLS默认的csim/build目录结构
    std::string test_file = rel_path + "in0_bf16.bin";
    std::ifstream f(test_file.c_str());
    if (f.good()) {
        std::cout << "Data path found: " << rel_path << std::endl;
        return rel_path;
    } else {
        std::cerr << "ERROR: Cannot find data file at relative path: " << test_file << std::endl;
        exit(1);
    }
}

int main() {
    // 1. 内存分配
    uint16* in0 = new uint16[DATA_SIZE];
    uint16* in1 = new uint16[DATA_SIZE];
    uint16* out = new uint16[DATA_SIZE];
    uint16* mask = new uint16[DATA_SIZE];
    uint16* golden_data[8]; // 支持8个算子
    for (int i = 0; i < 8; ++i) {
        golden_data[i] = new uint16[DATA_SIZE];
    }

    // 2. 加载数据
    std::cout << "Loading test data..." << std::endl;
    std::string data_path = get_data_path();
    if (!load_binary_data(data_path + "in0_bf16.bin", in0) ||
        !load_binary_data(data_path + "in1_bf16.bin", in1) ||
        !load_binary_data(data_path + "mask_bf16.bin", mask)) {
        std::cerr << "Fatal error: Unable to load one or more input data files." << std::endl;
        return 1;
    }
    
    for (int i = 0; i < 8; ++i) {
        std::string golden_file = data_path + "golden_out_config_" + std::to_string(i) + "_bf16.bin";
        if (!load_binary_data(golden_file, golden_data[i])) {
            std::cerr << "Fatal error: Unable to load golden data for Config " << i << " from " << golden_file << std::endl;
            return 1;
        }
    }
    std::cout << "All data loaded successfully." << std::endl;
    std::cout << "=== HLS Activation Accelerator Testbench ===" << std::endl;

    // 3. 执行测试
    double total_time = 0.0;
    int total_failures = 0;

    for (int config = 0; config < 8; ++config) {
        std::cout << "\n--- Testing Config " << config << " ---" << std::endl;
        
        // 为每个配置选择正确的输入并加载到加速器
        uint16* current_in0 = in0;
        uint16* current_in1 = (config == 2) ? mask : in1;
        activation_accelerator(current_in0, current_in1, out, STAGE_LOAD, config);

        // 执行计算
        auto t1 = std::chrono::high_resolution_clock::now();
        activation_accelerator(current_in0, current_in1, out, STAGE_COMPUTE, config);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        double elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_time += elapsed;
        std::cout << "Config " << config << " compute time: " << elapsed << " ms" << std::endl;

        // 取回结果
        activation_accelerator(current_in0, current_in1, out, STAGE_STORE, config);
        
        // 保存HLS输出以供调试
        std::string output_filename = data_path + "hls_output_config_" + std::to_string(config) + ".bin";
        save_binary_data(output_filename, out);
        std::cout << "Output results saved to: " << output_filename << std::endl;

        // 使用基于容限的方法比较结果
        int errors = compare_results_with_tolerance(out, golden_data[config]);
        if (errors == 0) {
            std::cout << "Config " << config << " test PASSED!" << std::endl;
        } else {
            std::cout << "Config " << config << " test FAILED with " << errors << " errors exceeding tolerance." << std::endl;
            total_failures++;
        }
    }

    // 4. 清理和总结
    std::cout << "\nTotal compute time for all configs: " << total_time << " ms" << std::endl;

    for (int i = 0; i < 8; ++i) delete[] golden_data[i];
    delete[] in0;
    delete[] in1;
    delete[] out;
    delete[] mask;

    if (total_failures > 0) {
        std::cerr << "\n=== Test FAILED: " << total_failures << " configurations had errors. ===" << std::endl;
        return 1; // 返回非零值表示失败
    } else {
        std::cout << "\n=== All Tests PASSED! ===" << std::endl;
        return 0; // 返回0表示成功
    }
}