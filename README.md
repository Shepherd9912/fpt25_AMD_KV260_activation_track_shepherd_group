# High-Throughput FPGA Accelerator for LLM Activation Functions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for a novel FPGA accelerator for seven key LLM activation functions (Softmax, LayerNorm, RMSNorm, SiLU, GELU, Add, Mul), targeting high throughput and exceptional precision on edge devices.

## ✨ Key Features & Innovations

Our design, implemented on the **AMD KV260 platform**, is based on an 8-way data-parallel architecture and features three key contributions:

1.  **Unified Norm Core:** A specialized datapath for both LayerNorm and RMSNorm reuses shared computational logic (sum of squares), reducing DSP slice utilization by over 50% compared to separate implementations.
2.  **Three-Stage Approximation Method:** A hybrid approach combining offline-optimized Minimax lookup tables with online dual-step Newton-Raphson refinement in a mixed-precision pipeline. This method robustly meets a stringent less than 1e-3 L2 error requirement.
3.  **Dependency-Breaking Accumulator:** A novel micro-architecture for reduction operations that resolves loop-carried dependencies, enabling a maximum-throughput initiation interval (II) of 1 across the entire design.

## 🚀 Performance Highlights

The accelerator achieves the following end-to-end system performance at its peak operational frequency of **250 MHz**.

| Function  | Latency (ms) | Throughput (T/s) | Bandwidth (MB/s) |
| :-------- | :----------- | :--------------- | :--------------- |
| LayerNorm | **0.8254**   | **1211.53**      | **227.16**       |
| Softmax   | 0.8447       | 1183.83          | 221.97           |
| RMSNorm   | 0.8476       | 1179.83          | 221.22           |
| SiLU      | 0.3107       | 3218.96          | 603.55           |
| GELU      | 0.3164       | 3160.74          | 592.64           |
| Add       | 0.3231       | 3095.43          | 580.39           |
| Mul       | 0.3295       | 3034.95          | 569.05           |

## 🛠️ How to Use

### 1. Full Build Flow (HLS Synthesis & Vivado Implementation)

Place the `kernel_hls` and `vivado` directories under the same parent directory. To run the complete flow from HLS C-simulation to final bitstream generation, execute the following command from the parent directory:

```bash
cd kernel_hls
vitis_hls -f run_hls.tcl && cd ../vivado && vivado -mode batch -source run_vivado.tcl
```

### 2. On-Board Deployment and Testing

Transfer the entire `on_board` directory to the Jupyter environment on the target KV260 board. Open the `activation_accelerator.ipynb` notebook. Restart the kernel, clear all cell outputs, and then execute the cells sequentially to run the on-board test and verify the results.

## 📁 Directory Structure

```
project_root/
├── kernel_hls/
│   ├── offline/
│   │   ├── generate_lut_final.py
│   │   ├── gen_test_and_golden_data.py
│   │   └── generate_testbench_data.py
│   ├── accelerator.cpp, .h
│   ├── accelerator_utils.h
│   ├── special_handling_logic.h
│   ├── lut_tables.h
│   ├── testbench.cpp
│   └── test_data.cpp, .h
├── on_board/
│   ├── activation_accelerator.bit, .hwh
│   ├── activation_accelerator.ipynb
│   ├── X_/Y_test_tensor_bf16.bin
│   └── refs/
└── vivado/
    └── run_vivado.tcl
```

---

### **Project File Explanation**

#### The `kernel_hls/offline/` Directory
- `gen_test_and_golden_data.py`: Generates test vectors and golden reference data required for on-board testing.
- `generate_testbench_data.py`: Generates test vectors and golden reference data for the HLS testbench, exporting them into `test_data.h`.
- `generate_lut_final.py`: Generates the Look-Up Tables (LUTs) for mathematical functions (exp, erf, reciprocal, rsqrt) using Minimax approximation.

> **Note:** The two data generation scripts are functionally equivalent. The testbench uses `.cpp/.h` files for easier debugging, while the on-board test uses a more standardized binary `.bin` format.

#### The `kernel_hls/` HLS Source Code
- `accelerator.cpp`: The main HLS top-level function.
- `accelerator.h / accelerator_utils.h`: Header files defining core data types and IP interface parameters.
- `lut_tables.h`: Contains the pre-computed LUTs generated offline.
- `special_handling_logic.h`: Defines the "Dataflow 2" logic dedicated to processing special numerical values.
- `test_data.cpp, test_data.h, testbench.cpp`: C++ testbench files for software-level simulation and verification.

#### The `on_board/` Directory
- `activation_accelerator.bit / .hwh`: The compiled FPGA bitstream and hardware handoff files.
- `activation_accelerator.ipynb`: The Jupyter Notebook for on-board testing.
- `X_/Y_test_tensor_bf16.bin`: Binary test vectors.
- `refs/`: Directory containing golden reference vectors for verification.

#### The `vivado/` Directory
- `run_vivado.tcl`: TCL script to build the Vivado project and generate the bitstream.

## 🧠 Core Architectural Concept: Dual-Dataflow Architecture

This design utilizes a "dual-dataflow" architecture to isolate high-performance computation from complex control logic, ensuring both robustness and maximum throughput.
- **Dataflow 1 (Main Pipeline):** A heavyweight, deeply pipelined path for standard numerical inputs, optimized for an II of 1.
- **Dataflow 2 (Special Handling Logic):** A lightweight, parallel path that analyzes the input row for special cases (NaN, Infinity, etc.). It generates a decision code that arbitrates the final output, ensuring correct behavior without stalling the main pipeline.

## 🔄 Data Generation for Reproducibility

To generate all required data files from scratch, first clone this repository. Then, run the following scripts from the appropriate directories:

1.  **On-Board Data:**
    ```bash
    cd project_root/kernel_hls/offline
    python3 gen_test_and_golden_data.py --outdir ../../on_board
    ```

2.  **HLS Kernel LUTs:**
    ```bash
    cd project_root/kernel_hls
    python3 ./offline/generate_lut_final.py
    ```

3.  **HLS Testbench Data:**
    ```bash
    cd project_root/kernel_hls/offline
    python3 generate_testbench_data.py --outdir ../
    ```

## 📝 Citation

If you find this work useful in your research, please consider citing the associated paper (details to be added).

This project can be found on GitHub at: [https://github.com/Shepherd9912/fpt25_AMD_KV260_activation_track_shepherd_group](https://github.com/Shepherd9912/fpt25_AMD_KV260_activation_track_shepherd_group)
