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
    \brief NVFP4 FP4→BF16 GEMM Benchmark for SM110 (NVIDIA Thor)

    Adapted from CUTLASS example 72a_blackwell_nvfp4_bf16_gemm.cu.
    Benchmarks block-scaled FP4→BF16 mixed-precision GEMM on Blackwell SM110a.
*/

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <fstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include "cute/tensor.hpp"

#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "helper.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using namespace cute;

// Helper to recast pointers for NVFP4 sub-byte types (matching 72a example pattern)
template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

// FP4 block-scaled GEMM configuration for SM100/SM110
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Tile shape for FP4 GEMM (matching CUTLASS example)
using MmaTileShape = Shape<_256,_256,_128>;
using ClusterShape = Shape<_2,_2,_1>;

// Epilogue collective builder for FP4 GEMM
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, 8,
    ElementD, LayoutDTag, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Mainloop collective builder for FP4 GEMM
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, 32,
    ElementB, LayoutBTag, 32,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Stride and Layout types
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

// Sm1xxBlkScaledConfig for SFA/SFB layout generation
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// HostTensors
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;

/* Read GPU clock from sysfs (cur_freq in kHz), fallback to 1575 MHz */
static double read_clock_ghz() {
    const char *sysfs_paths[] = {
        "/sys/devices/13a00000.gpu/devfreq/13a00000.gpu/cur_freq",
        "/sys/devices/platform/13a00000.gpu/devfreq/13a00000.gpu/cur_freq",
        "/sys/devices/graphics/devfreq/graphics/cur_freq",
        nullptr
    };
    for (int i = 0; sysfs_paths[i]; ++i) {
        std::ifstream ifs(sysfs_paths[i]);
        int64_t khz = 0;
        if (ifs >> khz) {
            return static_cast<double>(khz) / 1e6;
        }
    }
    return 1.575;
}

struct Options {
  bool help = false;
  bool json_output = false;
  int iterations = 50;
  int warmup = 5;
  uint64_t seed = 42;
  int m = 0, n = 0, k = 0; // 0 = run all default shapes
  int swizzle = 0;

  void parse(int argc, char* argv[]) {
    cutlass::CommandLine cmd(argc, const_cast<char const **>(argv));
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("swizzle", swizzle);
    if (cmd.check_cmd_line_flag("json")) { json_output = true; }
  }

  void print_usage(std::ostream &out) const {
    out << "bench_nvfp4_fp4_bf16\n\n"
        << "  Blackwell NVFP4→BF16 block-scaled GEMM benchmark.\n\n"
        << "Options:\n"
        << "  --help              Display this usage statement\n"
        << "  --m=<int>           M dimension (default: run all default shapes)\n"
        << "  --n=<int>           N dimension\n"
        << "  --k=<int>           K dimension\n"
        << "  --iterations=<int>  Profiling iterations (default: 50)\n"
        << "  --warmup=<int>      Warmup iterations (default: 5)\n"
        << "  --seed=<uint64>     Random seed (default: 42)\n"
        << "  --swizzle=<int>     Cluster rasterization swizzle\n"
        << "  --json              JSON output\n\n";
  }

  double gflops(double runtime_s) const {
    uint64_t flop = uint64_t(2) * m * n * k;
    return double(flop) / 1.0e9 / runtime_s;
  }
};

struct TimingStats {
  double min_ms, max_ms, mean_ms, median_ms, stddev_ms;
  double sem_ms;    // standard error of the mean
  double ci95_ms;   // 95% confidence interval half-width
  double cv_pct;    // coefficient of variation (%)
  int count;
  int trimmed;
  static TimingStats from(const std::vector<double> &times, bool trim = true) {
    TimingStats s{};
    if (times.empty()) return s;
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    int n = (int)sorted.size();
    s.min_ms = sorted.front();
    s.max_ms = sorted.back();
    s.count = n;
    s.median_ms = (n % 2) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    int lo = (trim && n >= 10) ? (int)(n * 0.05) : 0;
    int hi = (trim && n >= 10) ? n - lo : n;
    int t = hi - lo;
    if (t <= 0) { t = n; lo = 0; hi = n; }
    s.trimmed = n - t;
    double sum = 0;
    for (int i = lo; i < hi; ++i) sum += sorted[i];
    s.mean_ms = sum / t;
    double sq = 0;
    for (int i = lo; i < hi; ++i) sq += (sorted[i] - s.mean_ms) * (sorted[i] - s.mean_ms);
    s.stddev_ms = std::sqrt(sq / t);
    s.sem_ms = (t > 1) ? s.stddev_ms / std::sqrt(static_cast<double>(t)) : 0.0;
    s.ci95_ms = s.sem_ms * 1.96;
    s.cv_pct = (s.mean_ms > 0) ? (s.stddev_ms / s.mean_ms * 100.0) : 0.0;
    return s;
  }
};

struct BenchResult {
    double gflops;
    double tflops;
    double eff;
    bool valid;
};

/* Benchmark a single (M,N,K) shape with CUTLASS Gemm3x host reference verification */
BenchResult bench_shape(int M, int N, int K, double peak_tflops, const Options &options) {
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    LayoutA layout_A = make_layout(make_shape(M, K, 1), stride_A);
    LayoutB layout_B = make_layout(make_shape(N, K, 1), stride_B);
    LayoutC layout_C = make_layout(make_shape(M, N, 1), stride_C);
    LayoutD layout_D = make_layout(make_shape(M, N, 1), stride_D);

    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    block_A.reset(cutlass::make_Coord(size(layout_A)));
    block_B.reset(cutlass::make_Coord(size(layout_B)));
    block_C.reset(cutlass::make_Coord(size(layout_C)));
    block_D.reset(cutlass::make_Coord(size(layout_D)));
    block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

    cutlass::reference::host::TensorFillRandomUniform(block_A.host_view(), options.seed + 2021, 2.0, -2.0, 0);
    cutlass::reference::host::TensorFillRandomUniform(block_B.host_view(), options.seed + 2022, 2.0, -2.0, 0);
    cutlass::reference::host::TensorFillRandomUniform(block_C.host_view(), options.seed + 2023, 2.0, -2.0, 0);
    cutlass::reference::host::TensorFillRandomUniform(block_SFA.host_view(), options.seed + 2024, 2.0, -2.0, 0);
    cutlass::reference::host::TensorFillRandomUniform(block_SFB.host_view(), options.seed + 2025, 2.0, -2.0, 0);

    block_A.sync_device();
    block_B.sync_device();
    block_C.sync_device();
    block_SFA.sync_device();
    block_SFB.sync_device();

    Gemm gemm;
    auto arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { block_A.device_data(), stride_A,
          block_B.device_data(), stride_B,
          block_SFA.device_data(), layout_SFA,
          block_SFB.device_data(), layout_SFB },
        { {1.0f, 0.0f},
          block_C.device_data(), stride_C,
          block_D.device_data(), stride_D }
    };

    // RAII workspace allocation (auto-freed on scope exit)
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    gemm.initialize(arguments, workspace.get());
    for (int w = 0; w < options.warmup; w++) {
        gemm.run();
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  Kernel error: %s\n", cudaGetErrorString(err));
        return {0, 0, 0, false};
    }

    // --- Reference verification via CUTLASS Gemm3x (host side) ---
    // Run one more iteration to get the output for verification
    gemm.initialize(arguments, workspace.get());
    CUTLASS_CHECK(gemm.run());
    cudaDeviceSynchronize();

    block_D.sync_host();
    block_C.sync_host();

    // Reference computation via host-side Gemm3x (block-scaled path)
    {
        using namespace cute;
        Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
        Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
        Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
        Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);

        cutlass::reference::host::GettBlockScalingMainloopParams<
            ElementAccumulator,
            decltype(tensor_A),
            decltype(tensor_SFA),
            decltype(tensor_B),
            decltype(tensor_SFB)
        > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

        auto tensor_C = cute::make_tensor(block_C.host_data(), layout_C);
        auto tensor_D = cute::make_tensor(block_reference_D.host_data(), layout_D);

        cutlass::reference::host::GettBlockScalingEpilogueParams<
            ElementAccumulator,
            ElementAccumulator,
            ElementAccumulator,
            decltype(tensor_C),
            decltype(tensor_D)
        > epilogue_params{1.0f, 0.0f, tensor_C, tensor_D};

        cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
    }

    // Comparison
    bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());
    passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
    passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);

    if (!passed) {
        printf("    [VERIFICATION FAILED]\n");
    }

    // --- Profiling iterations ---
    gemm.initialize(arguments, workspace.get());
    std::vector<cudaEvent_t> start_ev(options.iterations);
    std::vector<cudaEvent_t> stop_ev(options.iterations);
    for (int i = 0; i < options.iterations; i++) {
        cudaEventCreate(&start_ev[i]);
        cudaEventCreate(&stop_ev[i]);
    }
    for (int i = 0; i < options.iterations; i++) {
        cudaEventRecord(start_ev[i]);
        gemm.run();
        cudaEventRecord(stop_ev[i]);
    }
    cudaDeviceSynchronize();

    std::vector<double> times(options.iterations);
    for (int i = 0; i < options.iterations; i++) {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_ev[i], stop_ev[i]);
        times[i] = (double)ms;
        cudaEventDestroy(start_ev[i]);
        cudaEventDestroy(stop_ev[i]);
    }

    TimingStats stats = TimingStats::from(times, true);

    double gflops = (double)M * N * K * 2.0 / (stats.mean_ms / 1000.0) / 1e9;
    double tflops = gflops / 1000.0;
    double eff = peak_tflops > 0 ? tflops / peak_tflops * 100.0 : 0;

    printf("    Avg: %.3f ms (min: %.3f, max: %.3f, stddev: %.4f, 95CI: \u00B1%.4f, CV: %.2f%%)\n",
            stats.mean_ms, stats.min_ms, stats.max_ms, stats.stddev_ms, stats.ci95_ms, stats.cv_pct);
    printf("    -> %10.0f GFLOPS  %8.1f TFLOPS  %6.1f%%  %s\n\n",
           gflops, tflops, eff, passed ? "✓ valid" : "✗ INVALID");
    fflush(stdout);

    return {passed ? gflops : 0, passed ? tflops : 0, passed ? eff : 0, passed};
}

int main(int argc, char* argv[]) {
    Options options;
    options.parse(argc, argv);

    if (options.help) {
        options.print_usage(std::cout);
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cuda_ver; cudaRuntimeGetVersion(&cuda_ver);
    std::string cutlass_commit =
#ifdef CUTLASS_COMMIT
        CUTLASS_COMMIT;
#else
        "unknown";
#endif

    double clockGhz = read_clock_ghz();
    double peak_tflops = 1032.0;  // NVIDIA stated dense peak for Thor SM110a

    if (!options.json_output) {
        printf("GPU: %s (sm_%d.%d) | SMs: %d | Clock: %.0f MHz | CUDA: %d.%d | CUTLASS: %s | Peak: %.0f TFLOPS\n",
               prop.name, prop.major, prop.minor, prop.multiProcessorCount,
               clockGhz * 1000, cuda_ver / 1000, (cuda_ver % 1000) / 10,
               cutlass_commit.c_str(), peak_tflops);
        printf("Iterations: %d (warmup: %d, seed: %lu)\n\n", options.iterations, options.warmup, (unsigned long)options.seed);
    }

    // If --m/--n/--k ALL provided (> 0), run single shape
    if (options.m > 0 && options.n > 0 && options.k > 0) {
        if (!options.json_output) {
            printf("  M=%d N=%d K=%d\n", options.m, options.n, options.k);
        }
        auto result = bench_shape(options.m, options.n, options.k, peak_tflops, options);
        if (options.json_output) {
            printf("{\"benchmark\":\"nvfp4_fp4_bf16\",\"m\":%d,\"n\":%d,\"k\":%d,"
                   "\"gflops\":%.0f,\"tflops\":%.1f,\"efficiency\":%.1f,"
                   "\"correctness\":\"%s\""
                   ",\"toolchain\":{\"cuda_version\":%d,\"cutlass_commit\":\"%s\"}"
                   "}\n",
                   options.m, options.n, options.k,
                   result.gflops, result.tflops, result.eff,
                   result.valid ? "passed" : "failed",
                   cuda_ver, cutlass_commit.c_str());
        }
        return 0;
    }

    // Otherwise, run default shape list
    // JSON: collect results
    std::ostringstream json_out;
    if (options.json_output) {
        json_out << "{\"device\":\"" << prop.name << "\",\"sm\":" << prop.major * 100 + prop.minor
                 << ",\"sm_count\":" << prop.multiProcessorCount
                 << ",\"clock_ghz\":" << std::fixed << std::setprecision(3) << clockGhz
                 << ",\"peak_tflops\":" << std::fixed << std::setprecision(1) << peak_tflops
                 << ",\"toolchain\":{\"cuda_version\":" << cuda_ver
                 << ",\"cutlass_commit\":\"" << cutlass_commit << "\"}"
                 << ",\"options\":{\"iterations\":" << options.iterations
                 << ",\"warmup\":" << options.warmup
                 << ",\"seed\":" << options.seed
                 << "},\"results\":[";
    }

    // --- Square shapes ---
    struct SqShape { int m; const char* label; };
    SqShape sq_shapes[] = {
        {1024, "1024"}, {2048, "2048"}, {4096, "4096"}
    };
    int n_sq = 3;

    if (!options.json_output) {
        printf("%-10s  %12s  %10s  %8s\n", "M=N=K", "GFLOPS", "TFLOPS", "Eff%%");
        printf("--------------------------------------------------------------------\n");
    }

    for (int si = 0; si < n_sq; si++) {
        int M = sq_shapes[si].m, N = M, K = M;

        if (!options.json_output) {
            printf("  M=N=K=%d\n", M);
            fflush(stdout);
        }

        auto result = bench_shape(M, N, K, peak_tflops, options);

        if (options.json_output) {
            if (si > 0) json_out << ",";
            json_out << "{\"category\":\"square\",\"m\":" << M << ",\"n\":" << N
                      << ",\"k\":" << K << ",\"gflops\":" << std::fixed << std::setprecision(0) << result.gflops
                      << ",\"tflops\":" << std::fixed << std::setprecision(1) << result.tflops
                      << ",\"efficiency\":" << std::fixed << std::setprecision(1) << result.eff
                      << ",\"correctness\":" << (result.valid ? "\"passed\"" : "\"failed\"") << "}";
        }
    }

    // --- Rectangular shapes ---
    struct RectShape { const char* name; int m, n, k; };
    RectShape rect_shapes[] = {
        {"M4096xN2048xK4096", 4096, 2048, 4096},
        {"M2048xN4096xK4096", 2048, 4096, 4096},
        {"M4096xN4096xK2048", 4096, 4096, 2048},
    };
    int n_rect = 3;

    if (!options.json_output) {
        printf("\n=== Rectangular Shapes (LLM Patterns) ===\n");
        printf("%-18s  %12s  %10s  %8s\n", "MxNxK", "GFLOPS", "TFLOPS", "Eff%%");
        printf("--------------------------------------------------------------------\n");
    }

    for (int ri = 0; ri < n_rect; ri++) {
        int M = rect_shapes[ri].m, N = rect_shapes[ri].n, K = rect_shapes[ri].k;

        if (!options.json_output) {
            printf("  %s\n", rect_shapes[ri].name);
            fflush(stdout);
        }

        auto result = bench_shape(M, N, K, peak_tflops, options);

        if (options.json_output) {
            json_out << ",";
            json_out << "{\"category\":\"rectangular\",\"name\":\"" << rect_shapes[ri].name
                      << "\",\"m\":" << M << ",\"n\":" << N << ",\"k\":" << K
                      << ",\"gflops\":" << std::fixed << std::setprecision(0) << result.gflops
                      << ",\"tflops\":" << std::fixed << std::setprecision(1) << result.tflops
                      << ",\"efficiency\":" << std::fixed << std::setprecision(1) << result.eff
                      << ",\"correctness\":" << (result.valid ? "\"passed\"" : "\"failed\"") << "}";
        }
    }

    if (options.json_output) {
        json_out << "]}";
        printf("%s\n", json_out.str().c_str());
    }

    return 0;
}

#else

int main(int argc, char* argv[]) {
    std::cerr << "This benchmark requires CUTLASS_ARCH_MMA_SM100_SUPPORTED (sm_100+)." << std::endl;
    return 0;
}

#endif
