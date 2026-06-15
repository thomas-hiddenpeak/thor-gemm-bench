/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief A native BF16 dense GEMM baseline benchmark for Blackwell SM100 using CUTLASS.

    Uses OpClassTensorOp with bfloat16_t elements to measure native BF16 tensor core
    throughput on SM100/SM110a. Serves as the reference baseline for comparing FP4
    block-scaled GEMM efficiency.

    Adapted from 70_blackwell_fp16_gemm with BF16 types and our benchmark output format.

    Usage:
      $ ./bench_bf16_cutlass --m=4096 --n=4096 --k=4096 --iterations=5

    Compile via:
      # Default (M256 N128 K64, C2x2x1):
      nvcc -std=c++17 -O3 -arch=sm_110a \
        -I $CUTLASS_DIR/include -I $CUTLASS_DIR/tools/util/include -I $CUTLASS_DIR/examples/common \
        --expt-relaxed-constexpr --expt-extended-lambda \
        bench_bf16_cutlass.cu -o bench_bf16_cutlass -lcudart

      # Custom tile/cluster (also via -D flags):
      nvcc -std=c++17 -O3 -arch=sm_110a \
        -DTILE_M=128 -DTILE_N=128 -DTILE_K=128 -DTILE_CLUSTERM=1 -DTILE_CLUSTERN=1 \
        -I $CUTLASS_DIR/include -I $CUTLASS_DIR/tools/util/include -I $CUTLASS_DIR/examples/common \
        --expt-relaxed-constexpr --expt-extended-lambda \
        bench_bf16_cutlass.cu -o bench_bf16_cutlass -lcudart
*/

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::bfloat16_t;                              // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                       // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;     // Memory access granularity (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::bfloat16_t;                              // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                    // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;     // Memory access granularity (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = float;                                           // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::ColumnMajor;                    // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;     // Memory access granularity (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                           // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                  // Operator class tag — native tensor cores

// Tile shape defaults with -D compiler flag override for search
// Recommended tiles for BF16 on SM100: M256xN128xK64 (default), M128xN128xK128, etc.
#ifndef TILE_M
#define TILE_M 256
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 64
#endif
#ifndef TILE_CLUSTERM
#define TILE_CLUSTERM 2
#endif
#ifndef TILE_CLUSTERN
#define TILE_CLUSTERN 2
#endif
#ifndef TILE_CLUSTERZ
#define TILE_CLUSTERZ 1
#endif

#define TILE_CAT_(a,b) a##b
#define TILE_CAT(a,b)  TILE_CAT_(a,b)

using MmaTileShape_MNK = Shape<TILE_CAT(_,TILE_M), TILE_CAT(_,TILE_N), TILE_CAT(_,TILE_K)>;
using ClusterShape_MNK = Shape<TILE_CAT(_,TILE_CLUSTERM), TILE_CAT(_,TILE_CLUSTERN), TILE_CAT(_,TILE_CLUSTERZ)>;

// Build the epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Build the mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Compose into a kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;                   // Default to ClusterLaunchControl (CLC) based tile scheduler

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool json_output;

  float alpha, beta;
  int iterations;
  int warmup;
  int m, n, k;
  int swizzle = 0;
  uint64_t seed = 42;

  Options():
    help(false),
    json_output(false),
    m(4096), n(4096), k(4096),
    alpha(1.f), beta(0.f),
    iterations(50),
    warmup(5),
    swizzle(0),
    seed(42)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("swizzle", swizzle);
    cmd.get_cmd_line_argument("seed", seed);
    if (cmd.check_cmd_line_flag("json")) {
      json_output = true;
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "bench_bf16_cutlass\n\n"
      << "  Blackwell BF16 dense GEMM — CUTLASS native tensor core baseline.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --swizzle=<int>             Cluster rasterization swizzle\n"
      << "  --iterations=<int>          Number of profiling iterations (default: 50)\n"
      << "  --warmup=<int>              Number of warmup iterations (default: 5)\n"
      << "  --seed=<uint64>             Random seed for initialization (default: 42)\n"
      << "  --json                      Output results as JSON\n\n";

    out << "\n\nCompile-time tile configuration (via -D):\n\n"
      << "  -DTILE_M=256 -DTILE_N=128 -DTILE_K=64  (default)\n"
      << "  -DTILE_CLUSTERM=2 -DTILE_CLUSTERN=2 -DTILE_CLUSTERZ=1  (default)\n\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "./bench_bf16_cutlass" << " --m=8192 --n=8192 --k=8192 --iterations=10\n"
      << "$ " << "./bench_bf16_cutlass" << " --m=4096 --n=4096 --k=4096 --json\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// GPU environment fingerprint
struct EnvInfo {
  std::string gpu_name;
  int sm_count;
  double clock_mhz;

  static EnvInfo probe() {
    EnvInfo info{};
    cudaDeviceProp props;
    int dev;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);
    info.gpu_name = props.name;
    info.sm_count = props.multiProcessorCount;
    // Try reading clock from sysfs (devfreq)
    const char *sysfs_paths[] = {
      "/sys/devices/13a00000.gpu/devfreq/13a00000.gpu/cur_freq",
      "/sys/devices/platform/13a00000.gpu/devfreq/13a00000.gpu/cur_freq",
      "/sys/devices/graphics/devfreq/graphics/cur_freq",
      "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
      nullptr
    };
    for (int i = 0; sysfs_paths[i]; ++i) {
      std::ifstream ifs(sysfs_paths[i]);
      int64_t khz = 0;
      if (ifs >> khz) {
        info.clock_mhz = static_cast<double>(khz) / 1000.0;
        break;
      }
    }
    if (info.clock_mhz <= 0) {
      // Default Thor frequency
      info.clock_mhz = 1575.0;
    }
    return info;
  }
};

/// Peak TFLOPS: BF16 tensor cores on SM110a have same throughput as FP4 tensor cores.
/// NVIDIA states 1,032 TF dense for Thor SM110a at 1575 MHz base clock.
inline double compute_peak_tflops(const EnvInfo &/*env*/) {
  return 1032.0;
}

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double min_runtime_ms;
  double max_runtime_ms;
  double median_runtime_ms;
  double stddev_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_ = 0, double min_ = 0, double max_ = 0,
    double median_ = 0, double std_ = 0, double gflops_ = 0,
    cutlass::Status status_ = cutlass::Status::kSuccess,
    cudaError_t error_ = cudaSuccess)
  :
    avg_runtime_ms(avg_), min_runtime_ms(min_), max_runtime_ms(max_),
    median_runtime_ms(median_), stddev_ms(std_), gflops(gflops_),
    status(status_), error(error_), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, options.seed + 2023);
  initialize_block(block_B, options.seed + 2022);
  initialize_block(block_C, options.seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}
  };

  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

bool verify(const Options &options) {
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha),
    ref_A,
    ref_B,
    ElementAccumulator(options.beta),
    ref_C,
    ref_D);

  // Wait for kernel to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

  return passed;
}

/// Per-iteration timing statistics
struct TimingStats {
  double min_ms, max_ms, mean_ms, median_ms, stddev_ms;
  int count;

  static TimingStats from(const std::vector<double> &times, bool trim = true) {
    TimingStats s{};
    if (times.empty()) return s;
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    int n = (int)sorted.size();
    s.min_ms = sorted.front();
    s.max_ms = sorted.back();
    s.count  = n;
    s.median_ms = (n % 2) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;

    // Trim top/bottom 5% for mean/stddev
    int lo = (trim && n >= 10) ? (int)(n * 0.05) : 0;
    int hi = (trim && n >= 10) ? n - lo : n;
    int t  = hi - lo;
    if (t <= 0) { t = n; lo = 0; hi = n; }
    double sum = 0;
    for (int i = lo; i < hi; ++i) sum += sorted[i];
    s.mean_ms = sum / t;
    double sq = 0;
    for (int i = lo; i < hi; ++i) sq += (sorted[i] - s.mean_ms) * (sorted[i] - s.mean_ms);
    s.stddev_ms = std::sqrt(sq / t);
    return s;
  }
};

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options, const EnvInfo &env)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Warmup iterations (not timed)
  for (int w = 0; w < options.warmup; ++w) {
    CUTLASS_CHECK(gemm.run());
  }
  cudaDeviceSynchronize();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop with per-iteration timing
  if (options.iterations > 0)
  {
    // Initialize ONCE outside the timer to avoid host overhead contamination
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Per-iteration CUDA events
    std::vector<cudaEvent_t> start_ev(options.iterations);
    std::vector<cudaEvent_t> stop_ev(options.iterations);
    for (int i = 0; i < options.iterations; ++i) {
      CUDA_CHECK(cudaEventCreate(&start_ev[i]));
      CUDA_CHECK(cudaEventCreate(&stop_ev[i]));
    }

    for (int iter = 0; iter < options.iterations; ++iter) {
      CUDA_CHECK(cudaEventRecord(start_ev[iter], 0));
      CUTLASS_CHECK(gemm.run());
      CUDA_CHECK(cudaEventRecord(stop_ev[iter], 0));
    }
    cudaDeviceSynchronize();

    std::vector<double> times(options.iterations);
    for (int i = 0; i < options.iterations; ++i) {
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev[i], stop_ev[i]));
      times[i] = (double)ms;
      CUDA_CHECK(cudaEventDestroy(start_ev[i]));
      CUDA_CHECK(cudaEventDestroy(stop_ev[i]));
    }

    // Compute statistics with 5% outlier trimming
    TimingStats stats = TimingStats::from(times, true);
    result.avg_runtime_ms    = stats.mean_ms;
    result.min_runtime_ms    = stats.min_ms;
    result.max_runtime_ms    = stats.max_ms;
    result.median_runtime_ms = stats.median_ms;
    result.stddev_ms         = stats.stddev_ms;
    result.gflops            = options.gflops(result.avg_runtime_ms / 1000.0);

    double tflops   = result.gflops / 1000.0;
    double peak_tf  = compute_peak_tflops(env);
    double peak_pct = (peak_tf > 0.0) ? (tflops / peak_tf * 100.0) : 0.0;

    if (options.json_output) {
      std::ostringstream j;
      j << std::fixed;
      j << "{" << std::endl;
      j << "  \"benchmark\": \"bf16_gemm_cutlass\"," << std::endl;
      j << "  \"m\": " << options.m << "," << std::endl;
      j << "  \"n\": " << options.n << "," << std::endl;
      j << "  \"k\": " << options.k << "," << std::endl;
      j << "  \"tile_mnk\": \""
        << TILE_M << "x" << TILE_N << "x" << TILE_K << "\"," << std::endl;
      j << "  \"cluster_mnk\": \""
        << TILE_CLUSTERM << "x" << TILE_CLUSTERN << "x" << TILE_CLUSTERZ << "\"," << std::endl;
      j << std::setprecision(3);
      j << "  \"runtime_ms\": {" << std::endl;
      j << "    \"mean\": " << result.avg_runtime_ms << "," << std::endl;
      j << "    \"min\": " << result.min_runtime_ms << "," << std::endl;
      j << "    \"max\": " << result.max_runtime_ms << "," << std::endl;
      j << "    \"median\": " << result.median_runtime_ms << "," << std::endl;
      j << "    \"stddev\": " << result.stddev_ms << std::endl;
      j << "  }," << std::endl;
      j << std::setprecision(1);
      j << "  \"gflops\": " << result.gflops << "," << std::endl;
      j << std::setprecision(3);
      j << "  \"tflops\": " << tflops << "," << std::endl;
      j << "  \"peak_tflops\": 1032.0," << std::endl;
      j << std::setprecision(1);
      j << "  \"peak_pct\": " << peak_pct << "," << std::endl;
      j << "  \"correctness\": \"" << (result.passed ? "passed" : "failed") << "\"," << std::endl;
      j << "  \"gpu\": {" << std::endl;
      j << "    \"name\": \"" << env.gpu_name << "\"," << std::endl;
      j << "    \"sm_count\": " << env.sm_count << "," << std::endl;
      j << std::setprecision(0);
      j << "    \"clock_mhz\": " << env.clock_mhz << std::endl;
      j << "  }" << std::endl;
      j << "}" << std::endl;
      std::cout << j.str();
    } else {
      std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
      std::cout << std::fixed << std::setprecision(3);
      std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms"
                << " (min: " << result.min_runtime_ms
                << ", max: " << result.max_runtime_ms
                << ", stddev: " << result.stddev_ms << ")" << std::endl;
      std::cout << std::setprecision(1);
      std::cout << "  GFLOPS: " << result.gflops << std::endl;
      std::cout << "  TFLOPS: " << std::setprecision(3) << tflops
                << " (Peak: 1032 TF, " << std::setprecision(1) << peak_pct << "%)" << std::endl;
    }
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.8 or higher Toolkit to run this example
  // and must have compute capability at least 100.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));

  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (props.major != 10 && props.major != 11 && props.major != 12 && props.major != 13) {
    std::cerr << "This example requires a GPU with compute capability 100a|f, 101a|f, or 103a|f)." << std::endl;
    return 0;
  }

  //
  // Probe environment (GPU name, SMs, clock from sysfs)
  //

  EnvInfo env = EnvInfo::probe();
  std::cout << "GPU: " << env.gpu_name
            << " | SMs: " << env.sm_count
            << " | Clock: " << std::fixed << std::setprecision(0) << env.clock_mhz << " MHz"
            << " | Peak: 1032 TF (dense @ 1575 MHz)" << std::endl;

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  run<Gemm>(options, env);
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
