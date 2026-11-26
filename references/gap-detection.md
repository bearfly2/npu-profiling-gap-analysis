# 空泡发现与量化

本文件回答：**怎么从 profiling 数据里把空泡找出来、量出来、排序，并排除假空泡。**

## 1. 两个观测尺度

### 1.1 宏观：`step_trace_time.csv` 的 Free

`step_trace_time.csv` 通常包含 `Step`、`Computing`、`Communication(Not Overlapped)`、`Free`、`Preparing` 等列（不同 CANN 版本列名略有差异）。口径：

```text
Computing%     = Σ Computing                      / Σ (Computing + Communication(Not Overlapped) + Free)
Communication% = Σ Communication(Not Overlapped)  / Σ 总时间
Free%          = Σ Free                           / Σ 总时间
```

分型阈值是启发式：

| 判定 | 阈值 | 含义 |
|---|---|---|
| Hostbound（下发/空泡） | `Free > 20%` | 设备等 CPU 下发、Python 开销、同步点、数据加载 |
| Computing | `Computing > 85%` | 设备计算占主导 |
| Communication | `Communication > 10%` | HCCL 集合通信占比高 |
| 无明显瓶颈 | 都不满足 | 按阶段和算子继续下钻 |

注意：
- `Free` 是 step 级平均值，**会掩盖局部空泡**：一个 300 ms 的预处理空档放在 5 s step 里只有 6%。
- 训练场景要先排除 dataloader 空泡：step 间周期性空闲、worker 处理时间低于等待间隔。
- 推理没有训练循环时，把一次完整 pipeline（或 denoise 循环）当作一个 step。

### 1.2 微观：`kernel_details.csv` 的 stream gap

在同一个 step 的同一条 stream 上，按 `Start Time(us)` 排序后：

```text
gap_us = next.start - (prev.start + prev.duration)
```

这是本技能的主口径。它能把“Free 总量”拆成一个个具体空泡：哪个 step、哪条 stream、前后是什么算子、持续多久。

**一条 stream 的 gap 不一定是真实设备空闲。** 多流场景下，一条流的空泡可能被另一条流覆盖。真实空泡要用**所有 stream 的任务区间并集**判断：并集之外的时间段才是设备真正无事可做。

## 2. 数据准备与完整性检查

| 检查项 | 要求 | 缺失处理 |
|---|---|---|
| `kernel_details.csv` | 含 `Step Id`、`Stream ID`、`Name`、`Type`、`Start Time(us)`、`Duration(us)` | 缺列则重采；不同 CANN 版本列名略有差异，以实际文件为准 |
| `step_trace_time.csv` | 至少覆盖 1 个完整 step | 缺失时用 kernel_details 自行聚合 |
| `trace_view.json` | CPU 侧归因需要 | 缺失时用阶段打点 + 代码搜索补证 |
| profiler 配置 | `ProfilerActivity.CPU + NPU`；建议 `Level1 + PipeUtilization`；CPU 定位建议 `with_stack=True` | 配置不足时先重采 |
| step 范围 | 明确 warmup / 稳态 step | 默认跳过第 1 个 step（编译/初始化） |

`kernel_details.csv` 的常见坑：
- `Step Id` 在通信行可能为空；`Stream ID` 可能为 `N/A`。
- `Start Time(us)`、`Duration(us)` 值可能带尾部空格或 tab，解析时要 `strip()`。
- `Wait Time(us)` 是 task 自身等待时间，**不是 gap**；不要拿它当空泡。
- `Type` 是算子类型（`ReduceMean`、`MatMulV3` 等），`Accelerator Core` 才是 `AI_CORE` / `AI_VECTOR_CORE` / `AICPU` / `COMMUNICATION` 这类执行单元标识。

## 3. 提取方法

### 3.1 脚本（推荐）

```bash
# 基础：按 stream 扫 gap，打印 Top 30
python scripts/find_gaps.py --prof-dir /path/to/prof/rank0 --min-gap-ms 1 --top 30

# 直接指定 csv
python scripts/find_gaps.py \
  --kernel-csv /path/to/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --steps 3,4 --min-gap-ms 1

# 同时算设备级真实空闲（多流场景）
python scripts/find_gaps.py --prof-dir /path/to/prof/rank0 --mode both

# 只看某个空泡指纹，例如 Cat -> Muls（要求两者是同一 stream 上相邻 task）
python scripts/find_gaps.py --prof-dir /path/to/prof/rank0 \
  --match-prev 'Cat' --match-next 'Muls'

# 阶段边界窗口：最后一个 Cat 结束 -> 第一个 Muls 开始（即使中间夹了别的算子）
python scripts/find_gaps.py --prof-dir /path/to/prof/rank0 \
  --steps 3,4 --window-prev 'aclnnCat_ConcatD' --window-next 'aclnnInplaceMuls'

# 导出 JSON / CSV 供后续报告
python scripts/find_gaps.py --prof-dir /path/to/prof/rank0 \
  --json gap_candidates.json --csv gap_candidates.csv
```

脚本输出：
- `stream_gaps`：按 `(Step Id, Stream ID)` 计算的相邻 task 间隙，含前后算子名、gap_ms、step 内相对起止时间。
- `device_gaps`：按 step 合并所有任务区间后计算的真实设备空闲段。
- `window_gaps`：使用 `--window-prev/--window-next` 时的阶段边界窗口（最后一个 prev 结束 → 第一个 next 开始）。
- `patterns`：按 `前算子 → 后算子` 聚合的指纹统计（出现次数、总 gap、中位数、p90、涉及 step）；`device_patterns` 对应设备级空闲段。

脚本参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--prof-dir` | — | prof 根目录，递归找 `kernel_details.csv`；多个 rank 时用 `--rank` 或直接给单个 rank 目录 |
| `--kernel-csv` | — | 直接指定 `kernel_details.csv`，与 `--prof-dir` 二选一 |
| `--rank` | — | 多 rank prof 目录下按路径中的 `rank<N>` 过滤 |
| `--steps` | 全部 | 逗号列表，如 `3,4,5` |
| `--streams` | 全部 | 逗号列表，如 `2,3` |
| `--min-gap-ms` | `1.0` | 最小 gap 阈值（毫秒） |
| `--top` | `20` | 打印 Top N |
| `--mode` | `stream` | `stream` / `device` / `both` |
| `--match-prev` / `--match-next` | — | 正则过滤前一/后一算子名（相邻 task gap） |
| `--window-prev` / `--window-next` | — | 阶段边界窗口：每 step 内“最后一个 prev 结束 → 第一个 next 开始”，即使中间夹了其他算子 |
| `--json` / `--csv` | — | 输出路径 |

### 3.2 手工 pandas 片段

```python
import pandas as pd

df = pd.read_csv("kernel_details.csv")
df = df.dropna(subset=["Step Id", "Stream ID", "Start Time(us)", "Duration(us)"])
df["end(us)"] = df["Start Time(us)"] + df["Duration(us)"]
df = df.sort_values(["Step Id", "Stream ID", "Start Time(us)"])

grp = df.groupby(["Step Id", "Stream ID"], sort=False)
df["prev_end(us)"] = grp["end(us)"].shift()
df["prev_name"] = grp["Name"].shift()
df["gap_ms"] = (df["Start Time(us)"] - df["prev_end(us)"]) / 1000

gaps = df[df["gap_ms"] >= 1.0].copy()
print(gaps[["Step Id", "Stream ID", "prev_name", "Name", "gap_ms"]]
      .sort_values("gap_ms", ascending=False).head(30))
```

### 3.3 设备级真实空闲（多流）

单条 stream 的 gap 只能说明“这条流没事干”。真实设备空泡要按 step 合并所有 stream 的任务区间：

```text
intervals = Σ [start_task, end_task]（按 start 排序）
merged = merge(intervals)           # 有重叠或相接的区间合并
idle_gaps = merged[i+1].start - merged[i].end
```

如果所有 stream 的并集覆盖了整个 step，则设备没有真实空闲，单个 stream 的 gap 可能只是并行流的正常错峰；这时优化目标不是“消空泡”，而是看关键路径有没有变短。

## 4. 空泡候选排序与指纹

每个候选至少记录：

| 字段 | 说明 |
|---|---|
| `step` | 出现在哪些 step；至少跨 2 个 step 才算可复现 |
| `stream` | Stream ID |
| `prev_name` / `next_name` | 前一个、后一个 task 名称 |
| `gap_ms` | 单次 gap 时长 |
| `occurrences` | 每 step 出现次数 |
| `stage` | 对齐到的阶段/unit（如 `preprocess`、`vae.full_video_encode`、`backward`） |
| `impact_ms_per_step` | `gap_ms × occurrences`，或按指纹聚合后的每 step 总 gap |
| `share` | `impact_ms_per_step / E2E_step_time` |

**排序建议**：先按 `impact_ms_per_step` 降序，再看复现性和定位确定性。单次偶发的大 gap（如首步、checkpoint）不等于稳定收益。

**空泡指纹**：`prev_name → next_name`。同一指纹在不同 step 重复出现，通常对应固定代码路径（例：`aclnnCat_ConcatD_ConcatD → aclnnInplaceMuls_Muls_Muls` 在 VAE 前出现 232.1/193.3 ms）。指纹也是改后验证的依据：优化后该指纹应消失或显著缩小。

## 5. 伪空泡排除（先做，避免白干）

| 伪空泡 | 特征 | 处理 |
|---|---|---|
| **打点同步制造的 gap** | 采集 profiling 时 P2/P3 打点仍开着；gap 紧邻 `synchronize` 或计时边界 | 关闭打点后重采（如 `NPU_TUNE_PROBE=0`） |
| **首步编译 / 初始化** | 只出现在 step 1；后面 step 没有 | 跳过 warmup，只看稳态 step |
| **profiler 起止开销** | 出现在采集窗口首尾；窗口外正常 | 只看 active 窗口内的完整 step |
| **step 边界 / barrier** | 出现在相邻 step 之间；常与 `dist.barrier` / allreduce 对齐 | 按 step 内统计，或用设备级并集确认是否真的空闲 |
| **多流覆盖** | 单条流有 gap，但另一条流在跑 | 看设备级并集；不要把并行错峰当空泡 |
| **`Wait Time` 误读** | 把 task 的 `Wait Time(us)` 当 gap | 只计算 `next.start - prev.end` |
| **CPU 采样缺失** | trace 里没有 CPU 事件就认为 CPU 空闲 | CPU 侧没有采样 ≠ CPU 没工作；补 `with_stack` 或阶段打点 |
| **通信行 `N/A` stream** | 通信 task 的 `Stream ID` 为 `N/A`，和计算 task 混在一起算 | 通信单独分析，不要混进 stream gap |

## 6. 输出 schema

`find_gaps.py --json` 输出结构：

```json
{
  "kernel_csv": "/path/to/kernel_details.csv",
  "mode": "both",
  "min_gap_ms": 1.0,
  "steps": ["3", "4"],
  "window_gaps": [
    {
      "step": "3",
      "stream": "2",
      "gap_ms": 232.1,
      "prev_end_ms": 1373.4,
      "next_start_ms": 1605.5,
      "prev_name": "aclnnCat_ConcatD_ConcatD",
      "next_name": "aclnnInplaceMuls_Muls_Muls",
      "prev_type": "ConcatD",
      "next_type": "Muls"
    }
  ],
  "stream_gaps": [
    {
      "step": "3",
      "stream": "2",
      "gap_ms": 232.1,
      "prev_end_ms": 1373.4,
      "next_start_ms": 1605.5,
      "prev_name": "aclnnCat_ConcatD_ConcatD",
      "next_name": "aclnnInplaceMuls_Muls_Muls",
      "prev_type": "ConcatD",
      "next_type": "Muls"
    }
  ],
  "device_gaps": [
    {
      "step": "3",
      "gap_ms": 180.0,
      "prev_end_ms": 1200.0,
      "next_start_ms": 1380.0,
      "prev_name": "...",
      "next_name": "..."
    }
  ],
  "patterns": [
    {
      "prev_name": "aclnnCat_ConcatD_ConcatD",
      "next_name": "aclnnInplaceMuls_Muls_Muls",
      "count": 2,
      "total_gap_ms": 425.4,
      "median_gap_ms": 212.7,
      "p90_gap_ms": 228.5,
      "steps": ["3", "4"],
      "streams": ["2"]
    }
  ]
}
```

## 7. 与阶段打点对齐

如果有 `[STAGE]` 打点或 `pipeline_breakdown.md`：
- 先用算子指纹和阶段边界对齐（例：`aclnnCat` 结束 → `aclnnInplaceMuls` 开始，落在 `WanVideoUnit_InputVideoEmbedder.process` 的 `preprocess_video` 和 `vae.full_video_encode` 之间）；
- 再用阶段耗时交叉验证：`InputVideoEmbedder - vae.full_video_encode ≈ 0.37 s`，其中 profiler 暴露出的 232 ms 是能在 NPU 时间线上看到的部分；
- 阶段打点的同步会放大绝对时间，**只用同口径相对比较**，不要拿 stage 绝对值和 profiler 绝对值混比。

## 8. 脚本自检（用已知案例验证）

如果手头有 Kairos/Wan 的这两份 profile，可以直接验证脚本是否正确：

```bash
# baseline：应输出 232.1 / 193.3 ms
python scripts/find_gaps.py --prof-dir <...>/overlap100_nocontig_bs16/rank0 \
  --steps 3,4 --min-gap-ms 1 \
  --window-prev 'aclnnCat_ConcatD' --window-next 'aclnnInplaceMuls'

# 优化后：应输出 2.4 / 2.1 ms
python scripts/find_gaps.py --prof-dir <...>/fastvideo_uint8_bs16/rank0 \
  --steps 3,4 --min-gap-ms 1 \
  --window-prev 'aclnnCat_ConcatD' --window-next 'aclnnInplaceMuls'
```

若数值不一致，先检查 `Step Id` 范围、`Stream ID` 是否包含 2、以及 CSV 是否来自同一次采集。
