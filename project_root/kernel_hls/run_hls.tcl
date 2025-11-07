# ===================================================================================
# Vitis HLS 自动化脚本 - FPT'25 竞赛官方流程对齐版
#
# 说明:
# 1. 将此脚本放置在你的 HLS 源文件（.cpp, .h）所在的目录中。
# 2. 确保你已经将 C++ 顶层函数名修改为 "activation_accelerator"。
# 3. 运行 Vitis HLS: vitis_hls -f run_hls.tcl
# ===================================================================================

# Step 1: 配置工程变量 (与 vivado 脚本和 PYNQ 脚本保持一致)
set PROJECT_NAME "activation_accelerator"
set TOP_FUNCTION "activation_accelerator" ;# <-- 已更新为与官方一致的顶层函数名
set SOLUTION_NAME "baseline"                ;# <-- 已更新为与官方一致的解决方案名
set FPGA_PART "xck26-sfvc784-2LV-c" 
set CLOCK_PERIOD 4.0                       ;# 10ns -> 100 MHz, 与官方示例一致

# 你的多文件管理方式是正确的，予以保留
set design_files [list \
    "accelerator.cpp" \
    "accelerator.h" \
    "lut_tables.h" \
    "special_handling_logic.h" \
    "accelerator_utils.h" \
]
set testbench_files [list \
    "testbench.cpp" \
    "test_data.h" \
    "test_data.cpp" \
]

# Step 2: 创建和配置工程
# -reset 会清除旧工程，确保全新开始
open_project -reset $PROJECT_NAME

# 设置顶层函数
set_top $TOP_FUNCTION

# 添加设计源文件
add_files ${design_files}

# 添加测试平台文件
add_files -tb ${testbench_files}

# Step 3: 创建和配置解决方案 (Solution)
# -flow_target vivado 是一个好习惯，会生成面向 Vivado IP 集成的输出
open_solution -reset $SOLUTION_NAME -flow_target vivado

# 设置目标 FPGA 器件和时钟约束
set_part $FPGA_PART
create_clock -period $CLOCK_PERIOD -name default

# Step 4: 执行完整的 HLS 流程
puts "====================================================="
puts "INFO: Running C-Simulation..."
puts "====================================================="
# 验证 C++ 算法的正确性
csim_design

puts "====================================================="
puts "INFO: Running Synthesis..."
puts "====================================================="
# 将 C++ 代码转换为 RTL
csynth_design

puts "====================================================="
puts "INFO: Running C/RTL Co-simulation..."
puts "====================================================="
# 验证 RTL 行为与 C++ 代码是否一致
cosim_design -rtl verilog

puts "====================================================="
puts "INFO: Exporting IP for Vivado..."
puts "====================================================="
# 导出 IP 核，格式为 Vivado IP Catalog
# 添加 -evaluate verilog 可以在导出时获取更准确的资源和时序评估
export_design -format ip_catalog -evaluate verilog

puts "====================================================="
puts "INFO: HLS script finished successfully."
puts "INFO: IP has been exported to '${PROJECT_NAME}/${SOLUTION_NAME}/impl/ip'"
puts "====================================================="

# 退出脚本
exit