---
name: npu-profiling-gap-analysis
description: >-
  Ascend NPU profiling 空泡（free / idle gap）分析与优化设计技能。用于从 torch_npu.profiler / msprof 产出的 timeline 与 kernel 数据中，按 stream 找出空泡，量化 gap 的时长、频率与占比，判断根因是通信等待、CPU/host 处理、依赖同步还是下发不足，反查到具体代码 file:line，并按“消除 → 提前 → 并行/重叠 → 加速”决策树设计方案，最后用 profiling 复测验证收益。触发场景：profiling 空泡、stream gap、Free 占比高、NPU 利用率周期性掉零、预处理/VAE 前空档、tokenizer/processor 在关键路径、DataLoader 空泡、通信等待、hostbound、kernel launch 间隙、CPU 格式转换、怀疑 CPU 处理拖慢 NPU。适用于 PyTorch + torch_npu 的训练与推理模型。不负责 profiling 采集本身，也不做单算子 kernel 内部优化。
---

# NPU Profiling 空泡分析与优化设计

本技能解决一个问题：**profiling 时间线上某段 stream 没有任务执行（空泡 / free / idle gap），这段空泡到底在等什么，对应哪段代码，应该怎么优化。**

目标不是“看一遍 profiler”，而是走完一条证据闭环：

```text
发现并量化空泡
  → 归因：通信等待 / CPU 处理 / 依赖同步 / 下发不足 / 数据加载
  → 代码定位：落到具体阶段、函数、file:line
  → 方案设计：消除 → 提前 → 并行/重叠 → 加速
  → 复测验证：gap 是否缩小、E2E 是否下降、精度是否保持
```

## 何时使用 / 不适用

**适用：**
- profiler timeline 上看到 stream 中有明显空白段，但不确定是通信还是 CPU 拖慢
- `step_trace_time.csv` 的 `Free` 占比高，或 NPU 利用率周期性掉零
- 推理 pipeline 中预处理 / tokenizer / VAE 等阶段前出现空档
- 训练 step 中 dataloader、H2D、梯度通信相关空泡
- 已经定位到某个阶段，但需要进一步落到代码 `file:line` 并设计改法
- 优化后需要验证空泡是否真的消失，而不是被挪到别处

**不适用：**
- 还没采集 profiling → 先走 `model-infer-profiling` 或对应采集技能
- 纯算子内部性能优化（tiling、指令流水、kernel 利用率）→ 走算子级优化流程
- 纯精度问题（性能正常但结果不对）→ 先做精度排查
- 模型尚未在 NPU 上跑通 → 先做最小跑通检查

**与 npu-model-tuning 的衔接**：本技能可独立使用，也可作为 `npu-model-tuning` P4（profiling 深度分析）与 P5（优化方案设计）的专项补充；验证过的案例可回写到 `experience/optimization-methods.jsonl`。

## 输入 / 输出契约

### 输入

| 输入 | 必需 | 说明 |
|---|---|---|
| prof 目录或 `kernel_details.csv` | 是 | 至少含 `kernel_details.csv`；推荐同时有 `step_trace_time.csv`、`trace_view.json` |
| 模型代码目录 | 是（代码定位阶段） | 用于把空泡映射到 `file:line` |
| 启动脚本 / 复测命令 | 推荐 | 用于改后同口径复测 |
| 阶段划分 / 已有打点 | 可选 | 已有 pipeline breakdown 或 `[STAGE]` 打点可直接复用 |

### 输出

所有产物写到目标代码目录或工作目录下的 `gap_analysis/<YYYYMMDD>/`（不要写进技能目录）：

```text
gap_analysis/<date>/
├── gap_candidates.json          # 脚本输出的空泡清单（step、stream、前后算子、gap_ms、pattern）
├── gap_analysis_report.md       # 分析报告（按 assets/gap-analysis-report-template.md）
├── attribution.md               # 归因证据（通信 / CPU / 依赖 / 调度 / 数据加载）
├── code_map.md                  # 空泡 → 阶段 → 函数 → file:line 映射
├── validation.md                # 改后复测：gap、阶段耗时、E2E、精度
└── prof/                        # 优化前后 profiling 产物或路径索引
```

## 核心概念

- **宏观 Free**：`step_trace_time.csv` 中 `Free` 列占总时间比例。`Free > 20%` 提示 Hostbound，但它是 step 级平均值，会掩盖局部空泡。
- **微观 stream gap**：同一 `Step Id` + `Stream ID` 上，相邻两个 task 之间的时间差：
  `gap = next.start − (prev.start + prev.duration)`。
- **设备级真实空泡**：多流场景下，一条 stream 的 gap 可能被另一条 stream 的任务覆盖。真实空泡 = 所有 stream 的任务区间并集之外的空闲段。
- **空泡指纹**：`前一算子名 → 后一算子名` 的组合。相同指纹在每个 step 重复出现，通常对应固定的代码路径。

## 训练 / 推理场景差异

| 维度 | 训练 | 推理 |
|---|---|---|
| 空泡常见位置 | DataLoader / H2D、forward preprocess、backward 通信、optimizer/EMA | 请求预处理 / tokenizer、VAE 前后处理、denoise/decode 循环、KV cache 拷贝、多请求调度 |
| 主要观测指标 | `s/step`、`data/forward/backward`、`Communication(Not Overlapped)` | E2E 时延、首 token / 单请求耗时、阶段耗时、并发下的 stream 占用 |
| 常用手段 | worker 预取、预处理下沉、batch encode、通信 overlap | 多流、异步 H2D、批处理、缓存 / 预计算、图模式 |
| 典型陷阱 | 打点同步制造空泡、dataloader rank skew 伪装通信 | profiler 本身改变调度、首请求编译、多请求互相掩盖空泡 |

## 五步工作流

每进入一个阶段，先读对应 reference。

### Step 0 · 准备与数据校验

1. 确认 profiling 场景：训练还是推理、模型代码目录、启动命令、step 范围、profiler 配置。
2. 校验数据完整性：`kernel_details.csv` 是否存在、是否含 `Step Id` / `Stream ID` / `Name` / `Type` / `Start Time(us)` / `Duration(us)`；缺失则先重采或说明口径。
3. **采集 profiling 时关闭 P2 打点**（如 `NPU_TUNE_PROBE=0`）。打点里的 `synchronize()` 会人为制造空泡，让 Free 虚高、误判成 Hostbound。
4. 若要做 CPU 侧定位，确认 profiler 打开 `ProfilerActivity.CPU`，尽量带 `with_stack=True`。
5. 读 `references/gap-detection.md`。

### Step 1 · 发现与量化空泡

1. 跑脚本：

   ```bash
   python scripts/find_gaps.py --prof-dir <prof目录> --min-gap-ms 1 --top 30

   # 阶段边界窗口：最后一个 Cat 结束 -> 第一个 Muls 开始（即使中间夹了别的算子）
   python scripts/find_gaps.py --prof-dir <prof目录> \
     --steps 3,4 --window-prev 'aclnnCat_ConcatD' --window-next 'aclnnInplaceMuls'
   ```

   脚本按 `(Step Id, Stream Id)` 扫相邻 task 间隙；`--mode device` 可额外计算设备级真实空闲；`--window-prev/--window-next` 用于阶段边界窗口测量。
2. 同时看宏观指标：`step_trace_time.csv` 的 `Computing` / `Communication(Not Overlapped)` / `Free` 占比。
3. 输出空泡候选表，每项至少记录：step、stream、前后算子、gap_ms、每步出现次数、占阶段/E2E 比例、空泡指纹。
4. **只认可复现的空泡**：至少跨 2 个 step 重复出现；排除首步编译、warmup、profiler 起止、打点同步、step 边界 barrier 等伪空泡（见 `references/gap-detection.md` §5）。
5. 读 `references/gap-detection.md`。

### Step 2 · 归因分型

对每个空泡候选，按以下顺序判断：

| 类型 | 关键证据 | 下一步 |
|---|---|---|
| **通信等待** | gap 与 `hcom_*` / `Task Type=COMMUNICATION` 对齐；`Communication(Not Overlapped)` 高；多 rank 某 rank 等待 | 区分通信慢 / 通信串行 / 慢卡 / 伪通信（rank skew、barrier） |
| **CPU/host 处理** | gap 期间 NPU 无任务但 CPU 线程忙；`trace_view.json` 有 Python/CPU op；阶段打点与 gap 时长对齐 | 定位具体 CPU 代码 |
| **依赖/同步** | `synchronize()`、`.item()`、`.cpu()`、`.numpy()`、Event/Stream wait、`dist.barrier` | 判断能否去同步或提前 |
| **下发/调度不足** | 小 kernel 多、launch 间隙短而密集、host enqueue 慢 | 环境变量 / 融合 / 图模式 |
| **数据加载/IO** | 周期性 step 间空泡；`data` 阶段高；worker 队列空 | DataLoader 调参 / 预处理下沉 |
| **真实无工作** | 所有 stream 同时空闲，无 CPU 忙、无通信等待 | 流程本身存在串行阶段，考虑流水化 |

> 通信 vs CPU 是首要分叉。**不要因为 gap 旁边有 HCCL 就判通信**，也不要因为 CPU 占用高就判 CPU——两者都可能只是结果。

### Step 3 · 代码定位（必须落到 file:line）

1. **锁定时序窗口**：记录 gap 前后 kernel 名和相对时间（例：`aclnnCat_ConcatD` 结束 → `aclnnInplaceMuls` 开始，gap 232.1 ms；timeline 绝对时间与脚本 step 内相对时间相差一个 step 起点偏移）。
2. **映射到阶段**：用已有阶段划分 / `[STAGE]` 打点，把 gap 归到具体 pipeline unit（例：`WanVideoUnit_InputVideoEmbedder.process` 的 `preprocess_video`，在 `vae.full_video_encode` 之前）。
3. **映射到源码**：在模型代码中搜索阶段函数、算子调用、常见 CPU 模式；记录函数名和 `file:line`。
4. **实锤确认**：在可疑代码前后加打点（前后 `synchronize`），或临时开关关闭该逻辑后复测；不要只凭“看起来像”。
5. **通信侧定位**：查看 HCCL task、rank timeline、`dist.all_reduce/all_gather/barrier` 调用点、DeepSpeed 通信配置。
6. 读 `references/attribution-and-localization.md`。

### Step 4 · 方案设计

按决策树选择手段，优先级从高到低：

```text
这项工作是否必要？
├─ 不必要 → 消除（Eliminate）
│    ├─ CPU 格式转换往返：PIL ↔ NumPy ↔ Tensor、permute/contiguous、无意义 cast
│    ├─ 冗余 tokenizer/processor、重复计算、重复拷贝
│    └─ 不必要的 synchronize / .item() / .cpu()
├─ 必要，但数据可以更早拿到 → 提前（Advance / Ahead-of-time）
│    ├─ 预处理、tokenizer/processor 下沉 DataLoader worker
│    ├─ 静态特征 / latent 离线缓存
│    └─ 权重预取（npu_prefetch）
├─ 必要且不能提前 → 并行/重叠（Overlap）
│    ├─ CPU 与 NPU：多流、异步 H2D、worker 池
│    ├─ 通信与计算：overlap_comm、通算融合、bucket 调整
│    └─ 小算子风暴：batch、融合、减少 launch
└─ 工作本身太重 → 加速（Accelerate）
     ├─ batch encode、向量化、融合算子
     └─ cast/normalize 从 CPU 移到 NPU
```

每项方案写清：证据、改动位置（`file:line`）、预期收益及依据、侵入程度、风险、回退方式。**预期收益只能写“预期/待验证”，不能写成结果。**

### Step 5 · 复测验证

1. 用同一场景、同卡、同 batch/步数、同 profiler 配置复测；采集时关闭打点。
2. 对比：空泡指纹是否消失或缩小、`Free%`、阶段耗时、E2E / step time。
3. 检查空泡是否被“挪走”：原来阶段的 gap 可能转移到下一个阶段，必须看设备级真实空闲和总时长。
4. 精度校验：数据路径改动（cast 位置、batch encode、预处理下沉）必须做输出级对比（cosine / loss / PSNR/SSIM，按模型类型选择）。
5. 用 `assets/gap-analysis-report-template.md` 出报告；回退开关必须验证有效。

## 归因速查表

| 现象 | 更可能是 | 先查 |
|---|---|---|
| gap 与 `hcom_allReduce` / `hcom_allGather` 对齐，`Communication(Not Overlapped)` > 10% | 通信等待 | `communication.json`、多 rank、DeepSpeed overlap 配置 |
| gap 期间 CPU 线程忙，Python stack 有 tokenizer / PIL / NumPy / `preprocess` | CPU 处理 | 阶段打点、代码中的 `.cpu()`/`.numpy()`/`Image.fromarray`/`np.stack` |
| gap 出现在每步固定位置，长度与 `[STAGE]` 某 unit 对齐 | CPU 或数据加载 | 该 unit 的调用链、worker 配置 |
| gap 前后是小 kernel，间隙短、数量多 | 下发/调度 | `TASK_QUEUE_ENABLE`、CPU 绑核、融合算子、图模式 |
| 所有 rank 同时等待，且某 rank 的 compute 明显偏低 | 慢卡 / rank skew | `msprof-analyze cluster -m slow_rank`、各 rank timeline |
| gap 前有 `aclnnSynchronize` / `.item()` / `.cpu()` 调用 | 同步点 | 代码搜索 `synchronize|\.item\(|\.cpu\(` |
| 训练 step 间周期性空泡，`data` 阶段高 | 数据加载 | `num_workers`、`prefetch_factor`、`persistent_workers`、`pin_memory` |

## 关键原则

- **空泡是入口，不是结论**：每个空泡都必须落到“谁在等”和“哪段代码”。
- **先排除伪空泡**：打点 `synchronize`、首步编译、profiler 开销、step 边界都可能制造假 gap。
- **stream gap ≠ 设备空闲**：多流时用任务区间并集判断真实空泡；单流场景可直接用 stream gap。
- **通信不等于通信慢**：barrier / rank skew / 数据不均常伪装成通信等待，先确认是否所有 rank 同时等。
- **先消除、再提前、再重叠、最后加速**：能删的工作不要并行，能提前的工作不要留在关键路径。
- **一次只改一个变量**：每个优化点单独测量、单独记录收益和精度影响；多个改动一起上无法归因。
- **改前有基线，改后有复测**：没有同口径对照的收益只能写“预期”。
- **保留回退开关**：所有改动默认可关闭，默认值保持原行为。
- **精度是硬门**：数据路径改动必须做输出级对比，回归即回滚。

## 文件索引（渐进披露）

| 文件 | 何时读 |
|---|---|
| `references/gap-detection.md` | Step 1：空泡定义、数据校验、脚本用法、伪空泡排除、输出 schema |
| `references/attribution-and-localization.md` | Step 2/3：通信 vs CPU vs 依赖/调度判据、kernel→阶段→`file:line` 定位流程 |
| `references/solution-design.md` | Step 4：消除/提前/并行/加速决策树、模式库、收益估算、风险清单 |
| `references/casebook.md` | 需要参考真实案例时：VAE 前 CPU 格式转换空泡、tokenizer 下沉 DataLoader、VAE batch encode、伪通信陷阱 |
| `assets/gap-analysis-report-template.md` | Step 5：输出报告模板 |
| `scripts/find_gaps.py` | Step 1：从 `kernel_details.csv` 按 stream/设备提取空泡并聚合空泡指纹 |
