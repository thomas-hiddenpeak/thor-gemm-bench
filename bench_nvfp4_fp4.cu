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
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM100 architecture.

    This example demonstrate a simple way to instantiate and run a blockscaled NVFP4 GEMM on the NVIDIA Blackwell SM100 architecture
    on NVIDIA Blackwell SM100 architecture. The kernel outputs quantized fp4 values with scale factors that be the input of another GEMM.

    Similar to 72a_blackwell_nvfp4_bf16_gemm, this kernel leverages:
    1. Blockscaled tcgen05.mma instructions.

    2. Per-SM memory called Tensor Memory (TMEM)

    3. The extended warp-specialized kernel design introduced in Hopper enabled by use of TMEM
    which allows us to decouple the execution of MMA and epilogue into separate warps.

    4. A new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution).

    Usage:

      $ ./examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm --m=2048 --n=2048 --k=2048
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
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
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
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::float_e2m1_t;                          // Element type for D matrix operand
using         ElementSFD  = cutlass::float_ue8m0_t;                         // Element type for SFB matrix operand
using         ElementC    = float;                                          // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
using         LayoutSFDTag = LayoutDTag;                                    // Layout type for SFD should be same as D matrix operand

constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementCompute      = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config — configurable via compiler -D flags
// Usage: -DTILE_M=256 -DTILE_N=128 -DTILE_K=256
//        -DTILE_CLUSTERM=2 -DTILE_CLUSTERN=2 -DTILE_CLUSTERZ=1
//        -DTILE_INPUTSF=16 -DTILE_OUTPUTSF=16

#ifndef TILE_M
#define TILE_M 256
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 256
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
#ifndef TILE_INPUTSF
#define TILE_INPUTSF 16
#endif
#ifndef TILE_OUTPUTSF
#define TILE_OUTPUTSF 16
#endif

/* Token-paste helper: TILE_CAT(_, TILE_M) → _256 (CUTLASS Shape alias) */
#define TILE_CAT_(a,b) a##b
#define TILE_CAT(a,b)  TILE_CAT_(a,b)

using MmaTileShape  = Shape<TILE_CAT(_,TILE_M), TILE_CAT(_,TILE_N), TILE_CAT(_,TILE_K)>;
using ClusterShape  = Shape<TILE_CAT(_,TILE_CLUSTERM), TILE_CAT(_,TILE_CLUSTERN), TILE_CAT(_,TILE_CLUSTERZ)>;

constexpr int InputSFVectorSize  = TILE_INPUTSF;
constexpr int OutputSFVectorSize = TILE_OUTPUTSF;

// D = alpha * acc + beta * C
//      With BlockScaleFactor generation.
using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD, LayoutSFDTag,
    ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,                      // Epilogue schedule policy
    FusionOperation
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto                              // Kernel schedule policy. Auto or using targeted scheduling policy
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

using FusionOp = typename Gemm::EpilogueOutputOp;
constexpr bool IsBlockScaleSupported = FusionOp::IsBlockScaleSupported;
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
using LayoutSFD = typename SfdOutputCfg::LayoutSF;

//
// Data members
//

/// Initialization
StrideA stride_A;
LayoutA layout_A;
LayoutSFA layout_SFA;
StrideB stride_B;
LayoutB layout_B;
LayoutSFB layout_SFB;
StrideC stride_C;
LayoutC layout_C;
StrideD stride_D;
LayoutD layout_D;
LayoutSFD layout_SFD;

// The HostTensors are only used for allocating memory on host and device, and transferring data between host and device
// Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
// Output Tensors
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> block_SFD;
// Reference Output Tensors
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;
cutlass::HostTensor<ElementSFD, cutlass::layout::PackedVectorLayout> block_reference_SFD;
// Matrix-wide normalization constant
cutlass::HostTensor<ElementCompute, cutlass::layout::PackedVectorLayout> block_Normconst;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

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
    m(1024), n(1024), k(1024),
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

    out << "72b_blackwell_nvfp4_nvfp4_gemm\n\n"
      << "  Blackwell NVFP4 GEMM using a Warp Specialized kernel.\n\n"
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

    out << "\n\nExamples:\n\n"
      << "$ " << "./examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

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
  int cuda_version;          // e.g. 13033 for CUDA 13.3
  std::string cutlass_commit;

  static EnvInfo probe() {
    EnvInfo info{};
    cudaDeviceProp props;
    int dev;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&props, dev);
    info.gpu_name = props.name;
    info.sm_count = props.multiProcessorCount;
    int cv; cudaRuntimeGetVersion(&cv); info.cuda_version = cv;
    info.cutlass_commit =
  #ifdef CUTLASS_COMMIT
    CUTLASS_COMMIT;
  #else
    "unknown";
  #endif
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

/// Peak TFLOPS: NVIDIA stated 1,032 TF (dense FP4) for Thor SM110a
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
  double ci95_ms;    // 95% confidence interval half-width
  double cv_pct;     // coefficient of variation (%)
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_ = 0, double min_ = 0, double max_ = 0,
    double median_ = 0, double std_ = 0, double ci95_ = 0, double cv_ = 0,
    double gflops_ = 0,
    cutlass::Status status_ = cutlass::Status::kSuccess,
    cudaError_t error_ = cudaSuccess)
  :
    avg_runtime_ms(avg_), min_runtime_ms(min_), max_runtime_ms(max_),
    median_runtime_ms(median_), stddev_ms(std_),
    ci95_ms(ci95_), cv_pct(cv_), gflops(gflops_),
    status(status_), error(error_), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed) {

  double scope_max, scope_min;
  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

  if constexpr (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  }
  else if constexpr (bits_input <= 6) {
    scope_max = 2;
    scope_min = -2;
  }
  else if constexpr (bits_input <= 8) {
    if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) {
      scope_max = 4;
      scope_min = 1;
    }
    else {
      scope_max = 1;
      scope_min = -1;
    }
  }
  else{
    scope_max = 4;
    scope_min = -4;
  }
  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  using namespace cute;
  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  // For SFD tensor layout
  using Sm1xxBlockScaledOutputConfig=  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
  layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
  layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFD = SfdOutputCfg::tile_atom_to_shape_SFD(cute::make_shape(options.m, options.n, options.k, 1));

  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_C.reset(cutlass::make_Coord(size(layout_C)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));
  block_Normconst.reset(cutlass::make_Coord(1));

  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  block_SFD.reset(cutlass::make_Coord(size(filter_zeros(layout_SFD))));

  initialize_block(block_A.host_view(), options.seed + 2021);
  initialize_block(block_B.host_view(), options.seed + 2022);
  initialize_block(block_C.host_view(), options.seed + 2023);
  initialize_block(block_SFA.host_view(), options.seed + 2024);
  initialize_block(block_SFB.host_view(), options.seed + 2025);
  block_Normconst.at(cutlass::make_Coord(0)) = 2;

  block_A.sync_device();
  block_B.sync_device();
  block_C.sync_device();
  block_D.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  block_SFD.sync_device();
  block_Normconst.sync_device();

}

// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    { // Mainloop arguments
      block_A.device_data(), stride_A,
      block_B.device_data(), stride_B,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    { // Epilogue arguments
      { options.alpha, options.beta },
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D}
  };

  if constexpr (IsBlockScaleSupported) {
    arguments.epilogue.thread.block_scale_factor_ptr = block_SFD.device_data();
    arguments.epilogue.thread.norm_constant_ptr      = block_Normconst.device_data();
  }

  arguments.scheduler.max_swizzle_size = options.swizzle;
  return arguments;
}

bool verify(const Options &options) {
  using namespace cute;
  // Create the arguments for host reference implementation
  Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
  Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
  Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);

  // think about how to simplify the gemm3x interface.
  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,                   // ElementAccumulator
      decltype(tensor_A),                   // TensorA
      decltype(tensor_SFA),                 // TensorSfA
      decltype(tensor_B),                   // TensorB
      decltype(tensor_SFB)                  // TensorSfB
    > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

  Tensor tensor_C = cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
  Tensor tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);
  Tensor tensor_SFD = make_tensor(block_reference_SFD.host_data(), layout_SFD);

  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementCompute,                       // ElementScalar
      ElementAccumulator,                   // ElementAccumulator
      ElementCompute,                       // ElementCompute
      decltype(tensor_C),                   // TensorC
      decltype(tensor_D),                   // TensorD
      decltype(tensor_SFD),                 // TensorSfD
      cute::Int<OutputSFVectorSize>,
      cutlass::reference::host::SfStrategy::SfDGen
    > epilogue_params {options.alpha, options.beta, tensor_C, tensor_D, tensor_SFD, block_Normconst.at(cutlass::make_Coord(0))};

  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // Comparison
  block_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());
  passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);

  block_SFD.sync_host();
  bool passed_sfd = cutlass::reference::host::TensorEquals(block_reference_SFD.host_view(), block_SFD.host_view());
  passed_sfd &= (cutlass::reference::host::TensorNorm(block_reference_SFD.host_view()) > 0);
  passed_sfd &= (cutlass::reference::host::TensorNorm(block_SFD.host_view()) > 0);

  return passed && passed_sfd;
}

/// Per-iteration timing statistics with 95% confidence interval and coefficient of variation
struct TimingStats {
  double min_ms, max_ms, mean_ms, median_ms, stddev_ms;
  double sem_ms;    // standard error of the mean
  double ci95_ms;   // 95% confidence interval half-width (1.96 * SEM for large n)
  double cv_pct;    // coefficient of variation (stddev/mean * 100)
  int count;
  int trimmed;      // number of trimmed outlier iterations

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
    s.trimmed = n - t;
    double sum = 0;
    for (int i = lo; i < hi; ++i) sum += sorted[i];
    s.mean_ms = sum / t;
    double sq = 0;
    for (int i = lo; i < hi; ++i) sq += (sorted[i] - s.mean_ms) * (sorted[i] - s.mean_ms);
    s.stddev_ms = std::sqrt(sq / t);
    // Standard error of the mean
    s.sem_ms = (t > 1) ? s.stddev_ms / std::sqrt(static_cast<double>(t)) : 0.0;
    // 95% CI half-width (z=1.96 for large n; acceptable for n >= 30)
    s.ci95_ms = s.sem_ms * 1.96;
    // Coefficient of variation
    s.cv_pct = (s.mean_ms > 0) ? (s.stddev_ms / s.mean_ms * 100.0) : 0.0;
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
    result.avg_runtime_ms   = stats.mean_ms;
    result.min_runtime_ms   = stats.min_ms;
    result.max_runtime_ms   = stats.max_ms;
    result.median_runtime_ms = stats.median_ms;
    result.stddev_ms        = stats.stddev_ms;
    result.ci95_ms          = stats.ci95_ms;
    result.cv_pct           = stats.cv_pct;
    result.gflops           = options.gflops(result.avg_runtime_ms / 1000.0);

    double tflops   = result.gflops / 1000.0;
    double peak_tf  = compute_peak_tflops(env);
    double peak_pct = (peak_tf > 0.0) ? (tflops / peak_tf * 100.0) : 0.0;

    if (options.json_output) {
      std::ostringstream j;
      j << std::fixed;
      j << "{" << std::endl;
      j << "  \"benchmark\": \"nvfp4_fp4_gemm\"," << std::endl;
      j << "  \"m\": " << options.m << "," << std::endl;
      j << "  \"n\": " << options.n << "," << std::endl;
      j << "  \"k\": " << options.k << "," << std::endl;
      j << std::setprecision(3);
      j << "  \"runtime_ms\": {" << std::endl;
      j << "    \"mean\": " << result.avg_runtime_ms << "," << std::endl;
      j << "    \"min\": " << result.min_runtime_ms << "," << std::endl;
      j << "    \"max\": " << result.max_runtime_ms << "," << std::endl;
      j << "    \"median\": " << result.median_runtime_ms << "," << std::endl;
      j << "    \"stddev\": " << result.stddev_ms << "," << std::endl;
      j << "    \"ci95\": " << result.ci95_ms << "," << std::endl;
      j << "    \"cv_pct\": " << result.cv_pct << std::endl;
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
      j << "  }," << std::endl;
      j << "  \"toolchain\": {" << std::endl;
      j << "    \"cuda_version\": " << env.cuda_version << "," << std::endl;
      j << "    \"cutlass_commit\": \"" << env.cutlass_commit << "\"" << std::endl;
      j << "  }" << std::endl;
      j << "}" << std::endl;
      std::cout << j.str();
    } else {
      std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
      std::cout << std::fixed << std::setprecision(3);
      std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms"
                << " (min: " << result.min_runtime_ms
                << ", max: " << result.max_runtime_ms
                << ", stddev: " << result.stddev_ms
                << ", 95%CI: \u00B1" << result.ci95_ms << " ms"
                << ", CV: " << result.cv_pct << "%)" << std::endl;
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
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
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
            << " | CUDA: " << (env.cuda_version / 1000) << "." << ((env.cuda_version % 1000) / 10)
            << " | CUTLASS: " << env.cutlass_commit
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
