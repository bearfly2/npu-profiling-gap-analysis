# 空泡归因与代码定位

本文件回答两个问题：**这段空泡在等谁？对应哪段代码？**

## 1. 归因总纲

先做首要分叉：**通信 vs CPU/host**。但不要停在二分，继续细分：

```text
空泡
├─ A. 通信等待
│   ├─ A1. 通信本身慢（HCCL kernel 时间长、带宽不足、消息碎片化）
│   ├─ A2. 通信与计算串行（未 overlap、stream 依赖、通信后置）
│   ├─ A3. 慢卡 / rank skew（某 rank 计算慢，其他 rank 等它）
│   └─ A4. 伪通信（barrier、数据不均、同步点，看起来像通信）
├─ B. CPU/host 处理
│   ├─ B1. Python 侧计算：tokenizer/processor、PIL/NumPy/OpenCV、逐样本循环
│   ├─ B2. 格式转换：PIL ↔ NumPy ↔ Tensor、permute/contiguous、CPU cast
│   ├─ B3. 数据加载 / IO：DataLoader worker、磁盘、解码、H2D
│   └─ B4. 框架开销：DeepSpeed/Accelerate/EMA/checkpoint/logging
├─ C. 依赖/同步等待
│   ├─ C1. 显式同步：`synchronize()`、`.item()`、`.cpu()`、`.numpy()`
│   ├─ C2. Stream/Event 依赖：等待前序 task 或另一条 stream
│   └─ C3. `dist.barrier()` / 迭代级 barrier
├─ D. 下发/调度不足
│   ├─ D1. 小算子过多，launch 间隙密集
│   ├─ D2. host enqueue 慢 / CPU 绑核不合理
│   └─ D3. 图模式未捕获 / 动态 shape 重编译
└─ E. 真实无工作
    └─ E1. 流程本身串行，设备确实无事可做
```

## 2. 通信型（A）

### 2.1 证据

- `kernel_details.csv` 中出现 `hcom_allReduce__*`、`hcom_allGather__*`、`hcom_broadcast__*`、`hcom_reduceScatter__*`、`hcom_alltoall__*`、`hcom_send/recv` 等；`Task Type` / `Accelerator Core` 可能标 `COMMUNICATION`。
- `step_trace_time.csv` 的 `Communication(Not Overlapped)` 占比 > 10%。
- `communication.json` / `communication_matrix.json` 可看到各通信域、rank、数据量、耗时。
- gap 与通信 task 的起止对齐；或 gap 后面紧跟一个通信 task（说明前面在等通信完成）。

### 2.2 细分判断

| 子类 | 证据 | 检查 |
|---|---|---|
| A1 通信本身慢 | HCCL kernel 本身耗时长；扩卡不线性；`Communication(Not Overlapped)` 随卡数上升 | 通信量估算、切分方式、HCCL_BUFFSIZE、网络/拓扑 |
| A2 通信与计算串行 | 通信 task 与计算 task 在时间上不重叠；`Overlapped=0`（如有该列） | 是否开启 `overlap_comm`、stream 依赖、通信是否后置到反向末尾、bucket 设置 |
| A3 慢卡 / rank skew | 个别 rank `Computing` 明显偏低、等待高；其他 rank 等它 | `msprof-analyze cluster -d ./profiling_data -m slow_rank -o ./result`；各 rank timeline；`npu-smi info` 看 HBM/温度/降频 |
| A4 伪通信 | 所有 rank 同时等；gap 与 `dist.barrier()` / 数据加载对齐；通信 task 本身很短 | 检查代码中的 `barrier`、DataLoader 各 rank 速度、样本长度分布、负载均衡 |

### 2.3 代码定位

搜索通信相关调用：

```bash
grep -rnE 'dist\.(all_reduce|all_gather|broadcast|reduce_scatter|all_to_all|barrier)|torch\.distributed' <repo>
```

DeepSpeed / Accelerate 配置重点：
- `overlap_comm`：梯度通信与反向计算重叠；
- `contiguous_gradients`：会额外做梯度 contiguous 拷贝，可能反而变慢；
- `reduce_bucket_size` / `allgather_bucket_size`：bucket 过小导致消息碎片化；
- `gradient_accumulation` / `bucket` 与模型规模是否匹配。

**重要**：通信优化不要一次改多个参数。`overlap_comm` 和 `contiguous_gradients` 经常一起出现，但收益可能主要来自其中一个，必须做隔离实验。

## 3. CPU/host 型（B）

### 3.1 证据

- gap 期间 NPU 没有 task，但 CPU 线程忙（`trace_view.json` 中 CPU op 连续）。
- Python stack（`with_stack=True`）显示 gap 对应 `tokenizer`、`processor`、`preprocess`、`PIL`、`numpy`、`DataLoader` 等函数。
- 阶段打点显示某个 unit 耗时与 gap 时长接近（例：`InputVideoEmbedder - vae.full_video_encode ≈ 0.37 s`，profiler 中 232 ms 空档）。
- 关闭/替换可疑 CPU 代码后，gap 明显缩小。

### 3.2 常见 CPU 热点清单

| 模式 | 典型代码 | 为什么产生 gap |
|---|---|---|
| tokenizer/processor 在关键路径 | `apply_chat_template`、`AutoProcessor`、`tokenizer(...)` | 纯 CPU，NPU 等待 |
| PIL/NumPy 往返 | `Image.fromarray`、`np.asarray`、`np.stack`、`cv2.resize` | 每样本多次转换和全量拷贝 |
| Tensor layout 转换 | `permute(...).contiguous()`、`.transpose()`、`.clone()` | 触发额外拷贝；在 CPU 上更慢 |
| CPU cast | `tensor.to(torch.bfloat16)` 在 H2D 之前 | CPU 上逐元素转换，且增加 H2D 数据量 |
| 逐样本循环 | `for i in range(batch): ...` 调 VAE/模型 | 小 kernel 多、launch 间隙多 |
| 不必要的 H2D/D2H | `.cpu()`、`.item()`、`.numpy()` | 强制同步 + PCIe 搬运 |
| 日志/EMA/checkpoint | 每步 `print`、`all_gather` loss、CPU EMA、save | 主进程阻塞，NPU 空等 |

### 3.3 代码搜索快捷命令

```bash
grep -rnE '\.cpu\(\)|\.item\(\)|\.numpy\(\)|synchronize\(\)' <repo>
grep -rnE 'Image\.fromarray|np\.asarray|np\.stack|np\.concatenate|cv2\.' <repo>
grep -rnE 'permute\(|\.contiguous\(\)|\.transpose\(' <repo>
grep -rnE 'tokenizer|processor|apply_chat_template' <repo>
```

先搜阶段函数内部，再向外扩到调用链；不要全局盲目改。

### 3.4 trace_view / MindStudio Insight 使用提示

- `trace_view.json` 是 Chrome trace 格式：事件含 `name` / `ts` / `dur` / `pid` / `tid` / `args`；NPU 流和 CPU 线程分开显示。
- 在 timeline 上框选 gap 区间，同时看对应的 CPU 线程：如果 CPU 线程在该区间持续有 `python` / `aten` / `preprocess` 事件，基本可判 CPU 型。
- 打开 `with_stack=True` 时，算子事件可能带 `python_stack`；直接搜 `tokenizer`、`processor`、`preprocess`、`Image.fromarray` 等关键字。
- MindStudio Insight 可直接框选 gap、查看调用栈和线程泳道；导出结论时保留截图和精确时间戳。
- 若没有 CPU trace，退化为“阶段打点 + 代码搜索 + 开关复测”，不要仅凭 timeline 空白就判 CPU 空闲。

## 4. 依赖/同步型（C）

### 4.1 证据

- gap 前或 gap 中有 `synchronize`、`.item()`、`.cpu()`、`.numpy()` 等强制同步调用。
- `trace_view.json` 中 CPU 在等 NPU（wait event），而不是在算。
- `dist.barrier()` / 迭代级 barrier 与 gap 对齐。
- 单卡没有该 gap、多卡才有 → 可能是 rank 间同步。

### 4.2 常见代码

```python
loss.item()                    # D2H 同步
x.cpu()                        # D2H 同步 + 拷贝
x.numpy()                      # NPU tensor 需先 D2H
torch.npu.synchronize()        # 全设备同步
dist.barrier()                 # 全 rank 同步
stream.wait_stream(other)      # stream 依赖
event.wait(stream)             # event 依赖
```

### 4.3 处理思路

- 能删则删：日志、调试、非必要同步。
- 不能删则改：累积到一定步数再同步、用 NPU 侧 tensor 计算、用 `torch_npu` 异步接口。
- `dist.barrier()` 若只是为计时对齐或“保险”，删除前确认无正确性依赖。

## 5. 下发/调度型（D）

### 5.1 证据

- kernel 数量极多、单个 kernel 很短（< 10–50 us），kernel 之间有密集小间隙。
- `api_statistic.csv` 中 enqueue/launch 相关 API 占比高。
- CPU 占用高但不一定是“重计算”，可能是在下发。
- 图模式 / `torch.compile` 未开启或频繁重编译。

### 5.2 处理方向

- 环境变量（按 CANN/torch_npu 版本实测）：`TASK_QUEUE_ENABLE=1/2`、`CPU_AFFINITY_CONF=2`。
- 融合算子：减少算子数量和 launch 次数。
- batch / 合并小算子：如 VAE 逐样本循环改 batch encode。
- 图模式：torchair / npugraph_ex（中侵入，需验证）。
- 绑核：避免跨 NUMA、避免 worker 与主进程争核。

## 6. 数据加载型（E）

### 6.1 证据

- 空泡出现在每个 step 开始/结束，周期性强。
- 训练日志 `data = next(data_iter)` 时间高；`[DATA_TIMING]` 显示 worker transform 时间长。
- `npu-smi` 显示设备利用率周期性掉零。
- DataLoader worker 数量不足 / 队列为空。

### 6.2 检查项

| 参数 | 说明 |
|---|---|
| `num_workers` | worker 太少 → 供数不足；太多 → CPU 超订 |
| `prefetch_factor` | 预取深度不足，尾延迟暴露到主进程 |
| `persistent_workers` | 每 epoch 重建 worker、重复加载 tokenizer/processor |
| `pin_memory` | 未开启时 H2D 是 pageable 同步拷贝，不能与计算重叠 |
| worker 线程数 | 多 worker 时 torch 线程池超订，需 `torch.set_num_threads(1)` |
| 数据格式 | 在 worker 里做重 CPU 处理（解码、resize），还是留给主进程 |

### 6.3 关键区分

- **`data` 时间短 ≠ 数据没问题**：DataLoader 可能在 step 之间预取，真正瓶颈是 worker 处理慢、队列供不上。
- **`data` 时间长 ≠ 一定加 worker**：如果 worker 已经很多、CPU 打满，加 worker 会更糟；应减少单样本处理量或把工作下沉到更早阶段。

## 7. 代码定位标准流程

### Step 1 · 锁定时序窗口

从 `find_gaps.py` 或 timeline 得到：

```text
step=3, stream=2
prev: aclnnCat_ConcatD_ConcatD, end=1338.6 ms
next: aclnnInplaceMuls_Muls_Muls, start=1570.7 ms
gap=232.1 ms
```

把绝对时间换算为 step 内相对时间，方便和代码打点对齐。

### Step 2 · kernel → operator → API → Python

| 层级 | 数据来源 | 方法 |
|---|---|---|
| kernel | `kernel_details.csv` 的 `Name` | 例：`aclnnCat_ConcatD_ConcatD` |
| operator | `operator_details.csv` / `op_statistic.csv` | 找同名 / 相近算子，确认是 `aten::cat` 还是别的 |
| API | `api_statistic.csv` | 找 `aten::cat`、`aten::mul_` 等 |
| Python | 代码搜索 + `trace_view.json` 的 stack | 搜 `torch.cat`、`mul_`、阶段函数 |

常见映射（仅供定位，不是绝对）：
- `aclnnCat_ConcatD*` → `torch.cat`
- `aclnnInplaceMuls*` / `aclnnMuls*` → `mul_` / `mul`
- `aclnnAdd*` → `add` / `+`
- `aclnnCast*` → `.to(dtype)` / `.float()` / `.half()`
- `aclnnTranspose*` / `aclnnPermute*` → `transpose` / `permute`
- `aclnnConcat*` / `ViewCopy` → `cat` / `contiguous` / copy

### Step 3 · 映射到 pipeline 阶段

- 如果已有 `pipeline_breakdown.md`，按其中 unit 顺序和调用点对齐。
- 如果没有，快速建一份：启动脚本 → pipeline `__call__/forward` → 各模型组件（text_encoder / DiT / VAE / preprocess）→ `file:line`。
- 用阶段打点（`[STAGE]` / `StageTimer`）与该 unit 的总耗时对照 gap。

### Step 4 · 落到 `file:line`

在阶段函数内搜索可疑模式，记录：

```text
阶段：WanVideoUnit_InputVideoEmbedder.process
调用点：aceworld_video_3_1_pre_wam.py:1675-1697
可疑函数：preprocess_video(input_video) 在 vae.full_video_encode 之前
具体实现：aceworld_video_3_1_pre_wam.py:1582-1637
根因：PIL → NumPy → np.stack ×2 → permute+contiguous → CPU cast → pageable H2D
```

### Step 5 · 实锤确认（三选一）

1. **打点确认**：在可疑代码前后加 `StageTimer`（前后 `torch.npu.synchronize()`），测出的耗时与 gap 对齐。测完移除打点，不进入最终 patch。
2. **开关确认**：临时关闭可疑路径（加环境变量开关），重采 profile，看 gap 是否消失。
3. **微基准确认**：把可疑 CPU 函数单独抽出来，用代表性输入跑 CPU 计时，确认其量级。

### Step 6 · 证据记录

每个空泡用统一格式记录：

```markdown
### GAP-001 · VAE 前 CPU 格式转换空泡
- 位置：step 3/4, stream 2
- 指纹：aclnnCat_ConcatD_ConcatD → aclnnInplaceMuls_Muls_Muls
- 时长：232.1 / 193.3 ms
- 阶段：WanVideoUnit_InputVideoEmbedder.process，vae.full_video_encode 之前
- 代码：aceworld_video_3_1_pre_wam.py:1582-1637（preprocess_video）、1675-1697（调用点）
- 根因：CPU PIL→NumPy→stack→permute→contiguous 格式转换往返
- 证据：stage 差 InputVideoEmbedder - vae.full_video_encode ≈ 0.37 s；关闭路径后 gap → 2.1/2.4 ms
- 方案：见 SOL-001
```

## 8. 常见误判

| 误判 | 实际情况 | 正确做法 |
|---|---|---|
| “有 HCCL 就是通信瓶颈” | 通信可能已经与计算重叠，只是存在 HCCL task | 看 `Communication(Not Overlapped)` 和通算时间线，不只看通信 task 是否存在 |
| “Free 高就是 CPU 慢” | 也可能是通信等待、慢卡、barrier、数据加载 | 先按 §1 分型，逐项找证据 |
| “CPU 占用高就是 CPU 处理瓶颈” | CPU 可能只是在提前下发/排队，NPU 并不缺任务 | 看 NPU 是否真的 idle、队列深度、gap 是否在 CPU 段 |
| “stream 2 有 gap，所以设备空着” | 其他流可能正在跑 | 用设备级并集判断真实空闲 |
| “VAE 慢，所以优化 VAE kernel” | gap 可能在 VAE 之前的 `preprocess_video`，根本不在 VAE 算子内 | 先看 gap 前后的 kernel 名和阶段边界，再决定优化对象 |
| “首步慢是模型问题” | 首步包含编译、内存池、算子初始化 | 跳过 warmup，只看稳态 |
| “多卡所有 rank 都在等通信” | 可能某一张卡慢导致其他 rank 等它 | 补采全 rank / `msprof-analyze cluster` 做慢卡检测 |
| “加 worker 一定更快” | CPU 已超订时加 worker 会更慢 | 先看 CPU 利用率、worker 队列、单样本处理量 |
