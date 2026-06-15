# FP4→FP4 Block-Scaled GEMM 优化规范

## 设备参数
- **GPU**: NVIDIA Thor (sm_11.0), 20 SMs
- **Clock**: 1,575 MHz (MAXN 模式)
- **FP4 Peak**: 1,032 TFLOPS (dense), ~2,064 TFLOPS (sparse)
- **计算**: 20 SMs × 32,768 FLOPs/cycle × 1.575 GHz = 1,032 TFLOPS
- **注意**: `cudaDevAttrClockRate` 在 Thor 上返回 1,049 MHz（不准确）。实际时钟以 `/sys/.../devfreq/cur_freq` 为准。

## 当前最佳
| 参数 | 值 |
|------|-----|
| ElementA/B | nv_float4_t<float_e2m1_t> |
| ElementC | float (input accumulation) |
| ElementD | float_e2m1_t (FP4 output) |
| ElementSFD | float_ue8m0_t (scale factor) |
| MmaTileShape | M256 × N128 × K256 |
| ClusterShape | C2 × C2 × C1 |
| SFVec | Input=16, Output=16 |

## 搜索结果

### Problem Size Scaling
| M=N=K | GFLOPS | TF | Peak % |
|-------|--------|-----|--------|
| 1024 | 60,426 | 60 | 5.8% |
| 2048 | 260,150 | 260 | 25% |
| 4096 | 493,703 | 494 | **48%** |
| 8192 | — | — | TBD |

### Tile Shape Matrix (M128 tiles)
| M×N | K | Cluster | GFLOPS | TF |
|-----|---|---------|--------|-----|
| 128×128 | 256 | C2×2 | 492,192 | 492 |
| 128×192 | 256 | C2×1 | 558,695 | 559 |
| 128×256 | 256 | C1×1 | — | smem error |

### Tile Shape Matrix (M256 tiles)
| M×N | K | Cluster | GFLOPS | TF |
|-----|---|---------|--------|-----|
| 256×128 | 256 | C2×2 | **576,000** | **576** |
| 256×192 | 256 | C2×1 | 569,133 | 569 |
| 256×192 | 256 | C2×2 | 526,557 | 527 |
| 256×256 | 256 | C2×1 | — | smem error |
| 256×256 | 256 | C2×2 | — | smem error |

### Cluster Shape (M128×128×K256, SFVec=16)
| Cluster | GFLOPS | TF | 状态 |
|---------|--------|-----|------|
| 2×2×1 | 492,046 | 492 | ✅ 最佳 (symmetric) |
| 4×1×1 | 287,183 | 287 | ✓ |
| 2×1×1 | 286,903 | 287 | ✓ |
| 1×2×1 | 286,625 | 287 | ✓ |
| 1×4×1 | 286,434 | 286 | ✓ |
| 1×1×1 | 231,245 | 231 | ✓ |
| 4×2×1 | — | — | ❌ |
| 2×4×1 | — | — | ❌ |
| 4×4×1 | — | — | ❌ |
| 2×2×2 | — | — | ❌ |

### SF Vector Size (M128×128×K256, C2×2×1)
| SFVec | GFLOPS | TF | 状态 |
|-------|--------|-----|------|
| 16 | 491,651 | 492 | ✅ 最佳 |
| 8 | 488,553 | 489 | ✓ |
| 32 | 480,249 | 480 | ✓ |
| 4 | 479,767 | 480 | ✓ |
| 64 | — | — | ❌ |

## 优化方向

### Phase 1: Cluster Shape ✅ DONE
- 已测试 10 种 cluster 组合
- **最佳**: C2×2×1

### Phase 2: SF Vector Size ✅ DONE  
- 已测试 SFVec 4, 8, 16, 32, 64 + asymmetric combos
- **最佳**: SFVec=16

### Phase 3: Tile Shape ✅ DONE (partial)
- 已测试 M128×N128 系列
- 已测试 M256×N192 和 M256×N128
- 最佳: M256×N128×K256 + C2×2×1 = 576 TF

### Phase 4: 下一步
1. ✅ M256×N128 + C2×2×1 = 576 TF (当前最佳)
2. 试 M256×N256 + C2×2×1 (smem error expected)
3. 试 M256×N64 + C2×1×1 (smaller N tiles)
4. 试 M512 系列 (if CUTLASS supports)
5. 手写 PTX kernel 突破 CUTLASS 限制

## 当前差距
- 576 TF (当前最佳) vs 826 TF (80% peak)
- 差距: 1.4×
