# 方案设计：消除 → 提前 → 并行/重叠 → 加速

本文件回答：**定位到代码之后，应该怎么改？**

## 1. 决策树（按优先级）

```text
已定位的空泡
│
├─ Q1. 这项工作本身是否必要？
│   ├─ 不必要 → 【消除 Eliminate】
│   │    ├─ CPU ↔ GPU/NPU 格式转换往返（PIL ↔ NumPy ↔ Tensor、无意义 cast）
│   │    ├─ 重复拷贝 / permute+contiguous / 重复 tokenizer
│   │    ├─ 非必要 synchronize / .item() / .cpu()
│   │    └─ 重复计算 / 重复 decode / 重复 encode
│   │
│   └─ 必要 → Q2. 输入数据是否本来就可以更早拿到？
│        ├─ 是 → 【提前 Advance / Ahead-of-time】
│        │    ├─ 预处理、tokenizer/processor 下沉 DataLoader worker
│        │    ├─ 首帧/静态特征预计算、离线缓存
│        │    ├─ 权重预取（npu_prefetch）
│        │    └─ 常量/元数据在初始化阶段算好
│        │
│        └─ 否 → Q3. 能否与其他工作重叠？
│             ├─ 能 → 【并行/重叠 Overlap】
│             │    ├─ CPU 与 NPU：多流、异步 H2D、worker 池
│             │    ├─ 通信与计算：overlap_comm、通算融合、bucket 调整
│             │    ├─ 小算子风暴：batch、融合、减少 launch
│             │    └─ 流水线阶段：chunked pipeline、double buffering
│             │
│             └─ 不能 → Q4. 工作本身能否更快？
│                  ├─ 能 → 【加速 Accelerate】
│                  │    ├─ batch encode、向量化、融合算子
│                  │    ├─ cast/normalize 从 CPU 移到 NPU
│                  │    └─ 数据格式/layout/dtype 优化
│                  │
│                  └─ 不能 → 记录为“结构性限制”，列风险与后续方向，不硬做
```

**核心原则：能删的工作不要并行，能提前的工作不要留在关键路径。**

## 2. 四类手段

### 2.1 消除（Eliminate）

**适用**：工作对结果没有贡献，或语义等价但多走了一圈。

典型场景：
- 数据在 dataset 里已经是 tensor，却又转 PIL、再转回 NumPy/Tensor；
- `permute(...).contiguous()` 后立刻又被 view/reshape；
- 为了用某个算子先 `.float()`、算完再 `.to(bf16)`；
- 同一份 tokenizer/processor 结果在每个 step 重复计算；
- 循环里每步 `.item()` / `.cpu()` 只为打日志。

实施要点：
1. 先证明语义等价（数值或输出级对比）；
2. 优先在数据源头改（dataset 输出正确格式），不要在下游打补丁；
3. 保留旧路径开关，便于回退和 A/B；
4. 注意边界：dtype 精度、数据顺序、随机性、异常输入。

**案例**：Kairos3.2 的 `preprocess_video` 把 [C,T,H,W] float tensor 转成 PIL list、再 144 次 `np.asarray`、两次 `np.stack`、`permute+contiguous`，最后又归一化回 bf16。语义上是恒等往返，直接输出 uint8 tensor + worker collate + NPU cast/normalize，VAE 前空泡 232.1/193.3 → 2.1/2.4 ms。

### 2.2 提前（Advance / Ahead-of-time）

**适用**：工作必要，但它的输入数据在更早阶段就已经可用，且不依赖当前 step 的中间结果。

典型场景：
- tokenizer / processor / chat template 在模型 forward 里执行 → 下沉到 DataLoader worker 的 collate；
- 视频/图像解码、resize、normalize → 放到 dataset `__getitem__` 或 worker；
- 首帧图像预处理、固定 prompt 编码 → worker 预计算；
- 静态特征 / latent / embedding → 离线缓存或内存缓存；
- 模型权重 → `torch_npu.npu_prefetch` 预取。

实施要点：
1. **输入是否真的更早可用**：如果预处理需要当前 step 的 DiT 输出，就不能提前；
2. **worker 安全**：DataLoader worker 里 fork 后不要持有 CUDA/NPU context；tokenizer/processor 要在 worker 初始化时创建并复用；
3. **顺序与随机性**：提前处理不能改变样本顺序、shuffle、随机增强的结果；
4. **资源**：worker 内存/CPU 会增加；多 worker 时控制 `torch.set_num_threads(1)`，避免线程池超订；
5. **DataLoader 参数配套**：`num_workers`、`prefetch_factor`、`persistent_workers`、`pin_memory`；
6. **回退**：保留模型内原始路径，worker 预计算失败时自动回退。

**案例**：Qwen `apply_chat_template + AutoProcessor` 原本在 `WanVideoUnit_PromptEmbedder` 主进程关键路径，0.328 s/step。下沉到 DataLoader worker 后 0.159 s/step；配 `prefetch_factor=4`、`persistent_workers=True`、`pin_memory=True`，`preprocess` 1.932→1.742 s，stage speed 5.29→4.851 s。

### 2.3 并行/重叠（Overlap）

**适用**：工作必要、不能提前，但与其他工作没有数据依赖，或依赖可以通过流水线拆开。

典型场景：
- CPU 预处理与 NPU 计算重叠：多流、独立 worker 线程、异步 H2D；
- 通信与计算重叠：DeepSpeed `overlap_comm`、`npu_mm_all_reduce_base`、通算融合；
- 小算子：batch 合并、算子融合、减少 launch；
- 多阶段流水：VAE / DiT / postprocess 分块流水，双缓冲；
- 权重加载与计算重叠：`npu_prefetch`。

实施要点：
1. **确认依赖**：没有数据依赖才可并行；有依赖时用流水线拆成多个 chunk；
2. **验证物理并行**：stream ID 分开不等于真并行，必须在 timeline 上用 `Start Time` / `Duration` 算 overlap；图模式下 GE 可能重排算子导致假并行；
3. **资源争抢**：CPU 线程、内存带宽、AI Core 可能互相抢；overlap 不一定降 wall；
4. **同步正确性**：跨流使用 `record_stream` / event / stream wait，避免悬垂指针；
5. **开关与回退**：保留单流/串行路径。

**案例**：DeepSpeed 反向通信 `overlap_comm=true` + `contiguous_gradients=false` + bucket 100MB，`Communication(Not Overlapped)` 1.074→0.766 s（−28.7%），backward 2.445→2.250 s（−8.0%）；但隔离实验显示主要收益来自关闭 `contiguous_gradients`，不是 overlap 本身。

### 2.4 加速（Accelerate）

**适用**：工作不可消除、不可提前、不可重叠，或已经在线但本身太慢。

典型场景：
- VAE 逐样本循环 → batch encode；
- Python 逐元素循环 → 向量化 / NPU 算子；
- 小算子多 → 融合算子；
- CPU cast → NPU cast；
- layout 转换多 → 统一 layout / 减少 transpose；
- 数据格式过大 → uint8 / bf16 / 压缩。

实施要点：
1. 先确认瓶颈占比，不要优化非热点；
2. 一次一个变量，单独测收益和精度；
3. batch encode 注意显存峰值和 cache 清理；
4. 精度变化必须验证（cast 位置、累加顺序、近似算子）。

**案例**：WanVideoVAE `batch_encode=False` 逐样本循环，`Conv3DV2` 3266 次调用，VAE encode free 288.8 ms（22.7%）。改 batch encode 后 `Conv3DV2` 206 次（−93.7%），free 46.8 ms（−83.8%），wall 1273.4→835.1 ms；kernel 本身只降 19.9%，说明主要收益来自减少 launch/搬运。

## 3. 模式库（症状 → 手段）

| # | 模式 | 典型症状 | 首选手段 | 关键风险 |
|---|---|---|---|---|
| P1 | CPU 格式转换往返 | PIL↔NumPy↔Tensor、`np.stack`、`permute+contiguous`、CPU cast | 消除：dataset 直接输出目标 tensor，worker collate，NPU cast | dtype/数值语义、T 不一致截断 |
| P2 | tokenizer/processor 在关键路径 | `apply_chat_template`、`AutoProcessor` 耗时与 gap 对齐 | 提前：下沉 DataLoader worker + 预取 | worker 初始化、顺序、内存 |
| P3 | 逐样本循环 / 小算子风暴 | 单 kernel 短、数量多、launch gap 密集 | 加速：batch encode / 融合 / batch | 显存峰值、cache 清理 |
| P4 | DataLoader 供数不足 | step 间周期性空泡、`data` 高、worker 队列空 | 提前 + 参数：`num_workers`/`prefetch`/`persistent_workers`/`pin_memory` | CPU 超订、长稳内存 |
| P5 | 同步点 | `.item()`/`.cpu()`/`synchronize()` 与 gap 对齐 | 消除：延迟/累积/去掉同步 | 日志语义、正确性 |
| P6 | 通信与计算串行 | HCCL 与计算不重叠、`Communication(Not Overlapped)` 高 | 重叠：`overlap_comm`、通算融合、bucket | 只改一个参数、bucket 太小 |
| P7 | 伪通信 / rank skew | 所有 rank 等、某 rank compute 低、barrier 与 gap 对齐 | 消除：去 barrier、均衡数据、修 dataloader | 等待可能转移到 allreduce |
| P8 | 权重加载卡顿 | 首层/首 step 空泡、权重 H2D 与计算串行 | 提前：`npu_prefetch`、权重预加载 | 依赖窗口、图模式 |
| P9 | 图模式 / 下发开销 | 小算子多、launch 密集、hostbound | 并行/加速：torchair / npugraph_ex、`TASK_QUEUE_ENABLE` | 图捕获失败、动态 shape |
| P10 | Python 主进程开销 | EMA/checkpoint/logging 与 gap 对齐 | 提前/异步：后台线程、减少频率、移出关键路径 | GIL、线程安全 |

## 4. 收益估算与优先级

### 4.1 预期收益

```text
单步收益 ≈ Σ(被消除/缩短的 gap_ms × 每步出现次数)
收益率   ≈ 单步收益 / 当前 E2E step_time
```

注意事项：
- 如果 gap 已被其他 stream 覆盖（设备级不空闲），收益上限要打折；
- 如果 CPU 工作与 NPU 计算本来就能重叠，收益来自“关键路径缩短”，不是“CPU 时间消失”；
- 优化后工作可能转移到其他阶段，必须用最终 E2E 验证；
- 不要拿历史数据做分母；基线必须当轮同卡重测。

### 4.2 优先级

按以下顺序排：
1. 收益大、侵入低、定位确定；
2. 收益大、侵入中，但可开关回退；
3. 收益中等、能与其他优化叠加；
4. 收益不明确或侵入高 → 只列候选，先不做。

每项方案标注：`预期收益（待验证）`、`侵入程度（低/中/高）`、`工作量`、`风险`、`回退方式`。

## 5. 方案卡片模板

```markdown
### SOL-XXX · <方案名>
- 对应空泡：GAP-XXX
- 根因：
- 改动位置：<file:line>
- 方案类型：消除 / 提前 / 并行重叠 / 加速
- 具体改动：
- 预期收益：<X ms/step，Y% E2E，待验证>
- 侵入程度：低 / 中 / 高
- 风险：
- 回退开关：
- 验证方式：<profile 对比 + 精度对比>
```

## 6. 风险检查清单

- [ ] 数据顺序 / shuffle / 随机增强是否改变？
- [ ] dtype / cast 位置是否改变数值结果？是否做了输出级对比？
- [ ] batch encode / 缓存是否增加显存或 host 内存？峰值是否可接受？
- [ ] worker 数量增加是否导致 CPU 超订？`torch.set_num_threads` 是否设置？
- [ ] fork worker 是否安全（无 NPU context、无锁、无未刷新文件）？
- [ ] 多流/异步 H2D 是否有正确的 stream/event 同步？tensor 生命周期是否安全？
- [ ] 通信配置是否改变 bucket/顺序？是否影响收敛或精度？
- [ ] 图模式是否支持当前动态 shape / 控制流？
- [ ] 默认开关是否保持原行为？回退路径是否验证过？
- [ ] 是否只改了一个变量？收益能否单独归因？

## 7. 实施顺序

1. **先固化基线**：同卡、同数据、同配置，至少 3 次，抖动 ≤ 10%。
2. **按方案卡片逐项实施**：一次一个优化点；每项单独编译/导入校验。
3. **每项单独测性能 + 精度**：有效且精度通过 → 固化；无效/负收益/精度回归 → 回滚并记录。
4. **全部叠加后做集成复测**：最终收益以串行集成结果为准，不能把分卡并行筛选的收益直接相加。
5. **回到 profiling 确认**：空泡指纹是否消失？是否出现新空泡？设备级 Free 是否下降？
6. **写报告与经验**：把“现象 → 根因 → 改动 → 收益 → 精度 → 风险”记录到 `gap_analysis_report.md`，可复用条目写入经验库或 `casebook.md`。
