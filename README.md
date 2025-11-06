# fpt25_AMD_KV260_activation_track_shepherd_group
A novel FPGA accelerator for seven key LLM activation functions, targeting high throughput and precision on edge devices

Our design, implemented on the AMD KV260 platform, is based on an 8-way data-parallel architecture and features three key contributions. First, a Unified Norm Core for both LayerNorm and RMSNorm reuses computational logic, reducing DSP slice utilization by over 50\%. Second, a three-stage approximation method, combining Minimax lookup tables with Newton-Raphson refinement in a mixed-precision pipeline, robustly meets a less than 1e-3 L2 error requirement. Third, a dependency-breaking accumulator micro-architecture enables a maximum-throughput initiation interval (II) of 1. 
