# NVFP4 GEMM on Jetson AGX Thor — 关键破局记录

## 结论

**NVFP4 在 Thor (sm_110) 上完全可用，ptxas 13.3 即可，无需特殊版本。**

实际性能（CUTLASS 72a sm_110a, ptxas 13.3）：

| Size | Runtime | GFLOPS | 验证 |
|---|---|---|---|
| 512³ | 0.008 ms | 32,345 | Passed |
| 1024³ | 0.015 ms | 138,612 | Passed |
| 2048³ | 0.083 ms | 207,872 | Passed |
| 4096³ | 0.604 ms | 227,409 | Passed |

理论峰值 ~1,035 TFLOPS → 4096³ 效率 22%。

---

## 关键破局点（3 个小时浪费在错的方向上）

### 破局点 1：编译 arch 必须是 `sm_110a` 不是 `sm_110`

```bash
-arch=sm_110      # ❌ 不启用 architecture-accelerated features
-arch=sm_110a     # ✅ 启用 TMA、tcgen05.mma.blockscaled、TMEM
```

`-a` 后缀是关键。`sm_110` 只生成基础指令集，不包含 Hopper/Blackwell 风格的加速指令。
CUTLASS 的 feature guard 都依赖 `__CUDA_ARCH_FEAT_SM110_ALL` 这个宏，而它只在 `sm_110a` 下定义。

参见：`/home/rm01/opencodeWorkspace/cutlass/include/cutlass/arch/config.h:137`

### 破局点 2：必须加 `--expt-relaxed-constexpr`

第一个编译失败的错误信息：
```
error: calling a constexpr __host__ function("get_utccp_smem_desc_tensor") 
from a __device__ function("mma_init") is not allowed.
```

这不是 ptxas 不支持 sm_110 NVFP4 指令 — 是缺少编译标志。`sm100_blockscaled_mma_warpspecialized.hpp:846` 用到了 host constexpr 函数，需要这个 flag 放宽限制。

完整编译命令：
```bash
nvcc -std=c++17 -O3 -arch=sm_110a \
  -I cutlass/include -I cutlass/tools/util/include -I cutlass/examples/common \
  --expt-relaxed-constexpr --expt-extended-lambda \
  bench_nvfp4_cutlass.cu -o bench_nvfp4_cutlass -lcudart
```

### 破局点 3：绕过 example 的软件 gate

`72a_blackwell_nvfp4_bf16_gemm.cu:514` 有个软件 gate 拒绝 sm_110：
```cpp
if (props.major != 10 || (props.minor != 0 && props.minor != 1 && props.minor != 3)) {
    std::cerr << "This example requires a GPU with compute capability 100a|f..." << std::endl;
    return 0;
}
```

绕过：扩展为接受 major 10/11/12。这是软件限制，不是硬件限制。

---

## 我之前的错误（记录下来避免重犯）

1. **瞎编 "ptxas 13.2/13.3 缺 sm_110 NVFP4 指令是回归"** — 完全错误。ptxas 13.3 完美支持。真实的失败是缺 `--expt-relaxed-constexpr` 和用了 `-arch=sm_110` 而非 `sm_110a`。

2. **没有直接做编译实验验证** — 浪费了大量时间在 web 调研和 Docker 拉取上。如果一开始就 `nvcc -arch=sm_110a --expt-relaxed-constexpr` 编译 CUTLASS 72a，30 分钟就能拿到结果。

3. **被自己的 "已知错误信息" 带偏** — 看到 "Feature not supported on sm_110" 就以为是 ptxas 不支持，没有怀疑是自己 PTX 语法写错了或编译标志不对。

---

## 完整可用文件

- `bench_nvfp4_cutlass.cu` — 修改过的 CUTLASS 72a，绕过 sm_110 software gate
- `build_nvfp4_cutlass.sh` — 一键编译脚本，包含所有必需 flag
- `bench_nvfp4_cutlass` — 编译产物

**编译**：`./build_nvfp4_cutlass.sh`
**运行**：`./bench_nvfp4_cutlass --m=4096 --n=4096 --k=4096 --iterations=20`

---

## 技术参考

- CUTLASS 4.5.2 `include/cute/arch/config.hpp:100-118` — 完整 SM110 feature 列表
- CUTLASS 4.5.2 `include/cutlass/arch/config.h:131-147` — `CUTLASS_ARCH_MMA_SM110_ENABLED` 定义
- CUTLASS 72a 原始文件：`cutlass/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu`
- 硬件：Jetson AGX Thor (T5000), sm_110, 20 SMs, 1.575 GHz boost, 122.9 GB GPU 内存
