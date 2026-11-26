# npu-profiling-gap-analysis

Ascend NPU profiling 空泡（free / idle gap）分析与优化设计 skill。

从 `torch_npu.profiler` / `msprof` 产出的 timeline 与 kernel 数据中：

1. 按 stream 找出空泡，量化 gap 的时长、频率与占比；
2. 判断根因是通信等待、CPU/host 处理、依赖同步还是下发不足；
3. 反查到具体代码 `file:line`；
4. 按“消除 → 提前 → 并行/重叠 → 加速”决策树设计方案；
5. 用 profiling 复测验证收益。

## 目录

```text
.
├── SKILL.md                                  # 技能入口：触发条件、五步工作流、决策树
├── references/
│   ├── gap-detection.md                      # 空泡定义、量化、脚本用法、伪空泡排除
│   ├── attribution-and-localization.md       # 通信 vs CPU vs 依赖/调度判据、代码定位
│   ├── solution-design.md                    # 消除/提前/并行/加速方案库
│   └── casebook.md                           # 已验证案例（VAE/tokenizer/VAE batch 等）
├── assets/
│   └── gap-analysis-report-template.md       # 分析报告模板
└── scripts/
    └── find_gaps.py                          # kernel_details.csv 空泡提取脚本
```

## 快速使用

```bash
# 扫描 kernel_details.csv 的 stream gap
python scripts/find_gaps.py --prof-dir /path/to/prof/rank0 --steps 3,4 --min-gap-ms 1 --top 30

# 阶段边界窗口：最后一个 Cat 结束 -> 第一个 Muls 开始
python scripts/find_gaps.py --prof-dir /path/to/prof/rank0 \
  --steps 3,4 --window-prev 'aclnnCat_ConcatD' --window-next 'aclnnInplaceMuls'

# 设备级真实空闲（多流）
python scripts/find_gaps.py --prof-dir /path/to/prof/rank0 --mode both
```

使用 Claude Code 时，将本目录放入 `.claude/skills/npu-profiling-gap-analysis`（或软链），然后调用 `/npu-profiling-gap-analysis`，或直接说“用 npu-profiling-gap-analysis 分析这个 profile”。

## 脚本依赖

仅 Python 标准库，无需额外安装。
