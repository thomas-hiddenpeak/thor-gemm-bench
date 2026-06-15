# CUDA Tile Benchmark — FP4 & BF16 GEMM on Blackwell (SM110)

[![CI](https://github.com/thomas-hiddenpeak/cuda-tile-benchmark/actions/workflows/benchmark-ci.yml/badge.svg)](https://github.com/thomas-hiddenpeak/cuda-tile-benchmark/actions/workflows/benchmark-ci.yml)

NVIDIA Jetson AGX Thor (sm_110a) 上的 NVFP4 和 BF16 Block-Scaled GEMM 系统性调优项目。基于 CUTLASS，覆盖 Tile Shape、Cluster 布局、SF Vector、Problem Size 等多维搜索。

## 硬件

| 参数 | 值 |
|---|---|---|
| GPU | NVIDIA Thor (sm_110a) |
| SM | 20 |
| Clock | 1,575 MHz (MAXN) |

## 当前最佳

| 类型 | Config | TFLOPS |
|---|---|---|---|
| **FP4→FP4** | M256×N128×K256 + C2×2×1 | **579** |
| BF16→BF16 (CUTLASS)`*` | M256×N128×K64 + C2×2×1 | 491 |
| BF16→BF16 (cuda_tile.h, ⚠️ 旧) | C2×2×1 | 458 |
| FP4→BF16 | SFD bottleneck | 464 |

`*` CUTLASS CollectiveBuilder (`OpClassTensorOp`, `bfloat16_t`) — 替代不可编译的 `bench_bf16.cu`。TF 数待实测确认。

**结论**：FP4→FP4 受 SFD（Scale Factor Decompression）瓶颈拖累，相比 BF16→BF16 有约 32% 额外开销。BF16 基线改用 CUTLASS 原生 BF16 tensor core 实现后性能待实测。CUTLASS 框架内继续调优的空间已不大，要进一步突破需手写 PTX 或架构级优化。

## 快速开始

### 依赖

- CUDA 13.3 (CUDA Toolkit 13.3)
- CUTLASS (master, 支持 `sm_110a`)

### 环境配置

编辑 `env.mk`（Makefile 环境）或 `env.sh`（shell 环境），设置本地路径：

```bash
# 方式一：直接编辑 env.mk
NVCC       ?= /usr/local/cuda-13.3/bin/nvcc
CUTLASS_DIR ?= /path/to/cutlass

# 方式二（推荐）：创建 env.local.mk，不会污染 git
$ cp env.mk env.local.mk
$ vi env.local.mk   # 修改 NVCC / CUTLASS_DIR
```

### 编译

```bash
cd cuda_tile_benchmark

# 编译所有 benchmark
make

# 或编译单个
make bench_nvfp4_fp4

# 自定义 tile/cluster（通过 Makefile pattern 规则）
make bench_nvfp4_fp4.m128n128 TILES="-DTILE_M=128 -DTILE_N=128 -DTILE_K=128"

# 或直接使用 shell 脚本（自动 source env.sh）
./run_tile_test.sh 256 128 256 2 1
```

> **注意**：`-arch=sm_110a` 必须带 `a` 后缀（启用 TMA/tcgen05.mma.blockscaled/TMEM），`--expt-relaxed-constexpr` 不能少。详见 [NVFP4_BREAKTHROUGH.md](NVFP4_BREAKTHROUGH.md)。

### 运行

```bash
./bench_nvfp4_fp4 --m=4096 --n=4096 --k=4096 --iterations=5
```

输出示例：
```
GPU: NVIDIA Thor | SMs: 20 | Clock: 2601 MHz | Peak: 1032 TF (dense @ 1575 MHz)
  Disposition: Passed
  Problem Size: 4096x4096x4096
  Avg runtime: 0.241 ms (min: 0.238, max: 0.246, stddev: 0.002)
  GFLOPS: 569805.7
  TFLOPS: 569.806
```

JSON 输出：
```bash
./bench_nvfp4_fp4 --m=4096 --n=4096 --k=4096 --json
```

### 命令行选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--m=N` | 1024 | M 维度 |
| `--n=N` | 1024 | N 维度 |
| `--k=N` | 1024 | K 维度 |
| `--iterations=N` | 50 | 性能测试迭代次数 |
| `--warmup=N` | 5 | 预热迭代次数（不计时） |
| `--seed=N` | 42 | 随机种子 |
| `--json` | off | 输出结构化 JSON |

## 项目结构

### 主要 Benchmark

| 文件 | 类型 | 说明 |
|---|---|---|---|
| `bench_nvfp4_fp4.cu` | **FP4→FP4** | 当前主 benchmark，CUTLASS CollectiveBuilder 实现 |
| `bench_nvfp4_fp4_bf16.cu` | FP4→BF16 | 混合精度（SFD 瓶颈分析） |
| `bench_bf16_cutlass.cu` | **BF16→BF16** | `OpClassTensorOp` + `bfloat16_t`，CUTLASS 原生 tensor core 基线 |
| `bench_bf16.cu` | BF16→BF16 | 基于 cuda_tile.h（⚠️ CUDA 13.3 不可编译，需 `-enable-tile`） |
| `legacy/bench_bf16_min.cu` | BF16→BF16 | 可运行的简化版（单 tile 配置） |
| `legacy/bench_nvfp4_cutlass.cu` | FP4→BF16 | CUTLASS 72a 移植（初始实验） |
| `legacy/bench_nvfp4_ptx.cu` | 手写 PTX | PTX 级别的 NVFP4 实验 |
| `legacy/bench_nvfp4_ultra.cu` | 实验性 | 超优化尝试 |

### 搜索脚本

通过编译器 `-D` 宏定义切换配置（不修改源码）。**每个配置编译约 1-2 分钟**（CUTLASS 模板实例化开销）。

| 脚本 | 用途 | 配置数 |
|---|---|---|---|
| `build_nvfp4_cutlass.sh` | CUTLASS 72a 移植编译 | 1 |
| `run_fp4_m256.sh` | M256 tile 系列 + Cluster | 6 |
| `run_fp4_search.sh` | M×N × SF Vector 全搜索 | 36 |
| `run_fp4_asymmetric.sh` | 非对称 tile (M128/M256) | 32 |
| `run_sf_search.sh` | SF vector size | 8 |
| `run_tile_search.sh` | 原始 tile shape × cluster | 40 |
| `run_tile_test.sh` | 单配置测试 | 1 |

结果自动保存到 `results/results_YYYYMMDD_HHMMSS.jsonl`。

### 辅助文件

| 文件 | 说明 |
|---|---|
| `env.mk` | Makefile 环境配置（NVCC、CUTLASS_DIR、PEAK_TFLOPS），支持 `env.local.mk` 覆写 |
| `env.sh` | Shell 环境配置（同上，供搜索脚本 `source` 使用） |
| `run_isolated.sh` | GPU 时钟锁定/MAXN 电源模式，确保可复现的 benchmark 条件 |
| `helper.h` | GpuTimer、CUDA/CUTLASS CHECK 宏 |
| `FP4_OPTIMIZATION_SPEC.md` | 完整优化规范与搜索记录 |
| `NVFP4_BREAKTHROUGH.md` | NVFP4 在 sm_110a 上的破局记录（踩坑指南） |
| `METHODOLOGY.md` | 测量方法学：计时方式、异常值处理、统计报告、已知限制 |
| `analyze_results.py` | 结果聚合工具：top-N 提取、分组统计、LaTeX 表格生成、CSV 导出、多实现对比 |
| `plot_results.py` | 发布级可视化：top-N 柱状图、scaling 折线图、tile×cluster 热力图、跨实现分组对比图 |
| `benchmark_suite.py` | 多运行 harness：N 次运行聚合（grand mean ± 95% CI）、JSON/CSV 报告输出 |

## 搜索结果

### FP4→FP4 Tile 搜索

| M×N | K | Cluster | TF | 状态 |
|---|---|---|---|---|
| 128×128 | 256 | C2×2×1 | 492 | ✅ |
| 128×192 | 256 | C2×1×1 | 559 | ✅ |
| 128×256 | 256 | C1×1×1 | — | ❌ smem |
| 256×128 | 256 | C2×2×1 | **579** | ✅ 最佳 |
| 256×192 | 256 | C2×1×1 | 570 | ✅ |
| 256×192 | 256 | C2×2×1 | 527 | ✅ |
| 256×256 | 256 | C2×1×1 / C2×2×1 | — | ❌ smem |
| 256×64 | 256 | C2×1×1 / C2×2×1 | — | ❌ hang |

### FP4→FP4 Cluster 搜索 (M128×128×K256)

| Cluster | TF |
|---|---|
| C2×2×1 | 492 ✅ |
| C4×1×1 / C2×1×1 / C1×2×1 / C1×4×1 | ~287 |
| C1×1×1 | 231 |
| C4×2×1 / C2×4×1 / C4×4×1 / C2×2×2 | ❌ 不可用 |

### BF16→BF16 Cluster 搜索 (C2×2×1)

| TF |
|---|
| 454 |
| 458 |

### Problem Size Scaling (M256×N128, C2×2×1)

| M=N=K | TF |
|---|---|
| 1024 | 60 |
| 2048 | 260 |
| 4096 | 579 |

完整搜索记录见 [FP4_OPTIMIZATION_SPEC.md](FP4_OPTIMIZATION_SPEC.md)。

## 目录结构

```
├── env.mk                      # Makefile 环境（集中管理 NVCC/CUTLASS_DIR）
├── env.sh                      # Shell 环境（同上，供搜索脚本 source）
├── run_isolated.sh             # GPU 时钟锁定/隔离 benchmark 工具
├── bench_nvfp4_fp4.cu          # FP4→FP4 主 benchmark
├── bench_nvfp4_fp4_bf16.cu     # FP4→BF16 混合精度
├── bench_bf16_cutlass.cu       # BF16→BF16 CUTLASS 原生 tensor core 基线
├── bench_bf16.cu               # BF16→BF16 基于 cuda_tile.h（旧基线，不可编译）
├── helper.h                    # GpuTimer、CUDA/CUTLASS CHECK 宏
├── build_nvfp4_cutlass.sh      # 编译脚本（CUTLASS 72a 移植）
├── run_fp4_*.sh                # 搜索脚本（通过 -D 宏，不修改源码）
├── run_sf_search.sh
├── run_tile_*.sh
├── FP4_OPTIMIZATION_SPEC.md    # 完整优化规范与搜索记录
├── NVFP4_BREAKTHROUGH.md       # sm_110a 踩坑记录
├── tile_search_results.md
├── analyze_results.py          # 结果聚合与分析工具
├── plot_results.py             # 发布级可视化工具
├── benchmark_suite.py          # 多运行聚合 harness
├── Makefile                    # 编译入口
├── README.md
├── METHODOLOGY.md              # 测量方法学文档
├── .gitignore
├── .editorconfig               # 编辑器格式统一配置
├── .github/workflows/          # CI 流水线配置
├── legacy/                     # 历史/实验性文件
├── probes/                     # 硬件探测工具
├── tests/                      # 测试文件
└── results/                    # 搜索结果（JSONL）+ README
```

## 关键发现

1. **SFD 是主要瓶颈**：FP4→FP4 的 Scale Factor 解压开销是主要性能限制因素，距 BF16 CUTLASS 基线仍有优化空间。
2. **BF16 基线已迁移**：`bench_bf16.cu`（基于 `cuda_tile.h`，不可编译）保留于根目录，新增 `bench_bf16_cutlass.cu`（CUTLASS `OpClassTensorOp` + `bfloat16_t`）为主推基线。
3. **C2×2×1 是最优 cluster**：FP4 和 BF16 均确认。C4×2×1 / C2×4×1 / C4×4×1 不可编译。
4. **M256 tile 提升有限**：M128→M128 到 M256→N128 提升约 18% (492→579 TF)，且 M256×N256 受 smem 限制。
5. **CUTLASS 内进一步调优空间有限**：579 TF 附近已接近 CUTLASS 框架上限，要突破需手写 PTX 或架构级优化。

## 下一步

已完成：
- [x] 路径集中管理：`env.mk` / `env.sh` 统一 NVCC/CUTLASS_DIR，支持 `env.local.mk` 覆写
- [x] 结果追踪：`results/*.jsonl` 仅忽略原始数据，元数据（README）可追踪
- [x] 搜索脚本通过 `-D` 编译器宏切换配置（不修改源码）
- [x] benchmark 输出结构化统计（per-iteration stats, min/max/stddev/median）
- [x] JSON 输出格式 + 结果自动归档
- [x] BF16 基线迁移到 CUTLASS 原生 tensor core（`bench_bf16_cutlass.cu`）
- [x] 目录重组（`legacy/`, `probes/`, `tests/`, `results/`）
- [x] 编译基础设施：Makefile + .editorconfig + .gitignore
- [x] 可复现 benchmark 工具：`run_isolated.sh`（GPU 时钟锁定）
- [x] 统计严格化：95% CI、CV、SEM 嵌入 benchmark 输出
- [x] CUTLASS commit + CUDA 版本编译时锁定（`-DCUTLASS_COMMIT`）
- [x] 测量方法学文档：`METHODOLOGY.md`（计时方式、异常值处理、已知限制）
- [x] 结果分析流水线：`analyze_results.py`（top-N / 分组 / LaTeX / CSV / 对比模式）
- [x] 发布级可视化：`plot_results.py`（柱状图、折线图、热力图、分组对比）
- [x] 多运行聚合：`benchmark_suite.py`（N 次运行 + grand mean ± 95% CI）
- [x] CI 流水线：`.github/workflows/benchmark-ci.yml`（lint + Python check + 编译验证）
- [x] 编辑器配置：`.editorconfig`（缩进 2 空格、LF 换行）

待完成：
- [ ] 手写 PTX kernel 绕过 CUTLASS 框架开销
- [ ] 探索 M512 / N512 系列
- [ ] SFD 流水线化 / 异步解压

## 踩坑记录

详见 [NVFP4_BREAKTHROUGH.md](NVFP4_BREAKTHROUGH.md) — 包含 3 个浪费时间的错误方向及正确解决方案。

## License

BSD-3-Clause (CUTLASS 示例代码)
