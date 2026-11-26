# 案例库

本文件记录已经验证过的空泡分析与优化案例，作为模式匹配和方案设计的参考。**案例是假设，不是结论**——遇到新模型时必须重新用 profiling 证据确认。

> 案例来源（Kairos3.2 / Wan2.2 优化记录）：
> - `docs/bs性能分析/20261124_预处理与预计算优化.md`
> - `docs/preprocess-uint8/README.md`
> - `docs/bs性能分析/bs8_bs16_阶段数据汇总.md`
> - `docs/bs性能分析/backward_optimization_20261123.md`
> - `docs/bs性能分析/a800_profiling_analysis_20261123.md`

## 案例索引

| 案例 | 空泡类型 | 根因 | 手段 | 关键收益 |
|---|---|---|---|---|
| C1 | VAE 前 CPU 格式转换空泡 | CPU PIL→NumPy→Tensor 往返 | 消除 | 232.1/193.3 → 2.1/2.4 ms |
| C2 | 预处理前 tokenizer/processor 空泡 | 纯 CPU 工作在关键路径 | 提前 | 0.328 → 0.159 s/step |
| C3 | VAE 逐样本循环 launch gap | 小 kernel / launch 过多 | 加速（batch） | free 288.8 → 46.8 ms |
| C4 | 伪通信：barrier / dataloader rank skew | 数据不均导致全 rank 等待 | 消除 / 均衡 | barrier 652 → 目标 145 ms |
| C5 | 通信优化归因陷阱 | `contiguous_gradients` 而非 `overlap_comm` | 隔离变量 | backward −8%，主要来自 contiguous=false |

---

## C1 · VAE 前 CPU 格式转换空泡（Kairos3.2 / Wan2.2）

**背景**：8×910B3（64 GB），BS16/rank（global 128），DeepSpeed ZeRO-1，gradient checkpointing，训练场景。

**现象**：
- profiler 时间线上，`aclnnCat_ConcatD_ConcatD` 结束于 1338.6 ms，`aclnnInplaceMuls_Muls_Muls` 开始于 1570.7 ms，中间 **232.1 ms** 无 kernel 执行。
- step 4 同样出现，gap **193.3 ms**。
- 位置在 `WanVideoUnit_InputVideoEmbedder.process` 内、`vae.full_video_encode` 计时开始之前。
- 关键判断：**这段不是 VAE 算子，也不是 VAE launch gap**；VAE 小算子融合、删除 Cast、调整 VAE batch encode 都覆盖不到它。

**定位过程**：
1. 用 `kernel_details.csv` 锁定 gap 前后 kernel：`aclnnCat_ConcatD` → `aclnnInplaceMuls`。
2. 与 stage 打点交叉验证：`InputVideoEmbedder - vae.full_video_encode ≈ 0.37 s`，其中 profiler 暴露出的 232 ms 是 NPU 时间线上可见的部分。
3. 顺调用链定位到 `aceworld_video_3_1_pre_wam.py:1582-1637` 的 `preprocess_video`，调用点在 `1675-1697`，位于 `vae.full_video_encode` 之前。
4. 确认根因：FastWAM dataset 返回的 `video` 已经是 `[C,T,H,W]`、`[-1,1]` float tensor，训练路径又做了一次恒等往返：
   `denormalize_video → uint8 → permute [T,H,W,C] → 每样本 9 次 Image.fromarray → PIL list → collate → 每样本 9 次 np.asarray → np.stack ×2 → torch.from_numpy().permute(0,4,1,2,3).contiguous() → CPU uint8→bf16 → pageable H2D → mul_/add_`。
   其中 144 次 `np.asarray`、两次 `np.stack`、一次 `permute+contiguous` 是主要 CPU 开销。

**方案（消除）**：
- dataset 的 `denormalize_video(..., return_tensor=True)` 保留同一份 `[C,T,H,W]` uint8 tensor；
- DataLoader worker 内 collate `torch.stack` 成 `[B,C,T,H,W]`；
- `preprocess_video` 走 tensor fast path：uint8 H2D（约 49.5 MB/batch）→ NPU cast bf16 → `mul_/add_` 归一化；
- `pin_memory=True`，使 H2D 可异步；
- 保留 PIL list 供 `PromptEmbedder` / `ImageEmbedderFused` / bucket 元数据使用。

**结果**：
- VAE 前空泡 **232.1 / 193.3 ms → 2.1 / 2.4 ms**（基本消失）。
- `InputVideoEmbedder` 0.901 → 0.703 s（−22.0%）。
- 30-step timing-off `speed` 5.563 → 5.090 s（−8.5%），`forward` 2.800 → 2.630 s（−6.1%）。
- 12-step loss 与 baseline 完全一致（step3 84.6423、step4 131.3471）。
- 收益拆解：H2D 传输量减少约 10–20 ms；CPU cast 省几十 ms；**CPU 格式转换和拷贝约 150–200 ms，是主要收益**。

**经验**：
- 空泡在 VAE 前，不代表要优化 VAE kernel；先看 gap 前后 kernel 和阶段边界。
- “语义恒等往返”是最值得优先消除的一类 CPU 工作。
- 数据格式在 dataset 源头决定，下游打补丁收益有限。

---

## C2 · 预处理前 tokenizer/processor 空泡（Kairos3.2 / Wan2.2）

**背景**：同 C1，训练场景，BS16/rank。

**现象**：
- `WanVideoUnit_PromptEmbedder` 中的 Qwen processor/tokenizer 工作（`apply_chat_template` + `AutoProcessor` 处理 prompt 和首帧图像）是纯 CPU 工作，原先在训练主进程的 forward 关键路径上执行。
- NPU 需要等这段 CPU 处理完成才能进入文本编码；BS16 下 `tokenizer/processor` 约 **0.328 s/step**。
- gap 出现在预处理阶段之前/内部，属于典型的“NPU 等 CPU”。

**定位过程**：
- 阶段打点显示 `PromptEmbedder`（不含 vision）0.834 s，其中 `tokenizer/processor` 0.328 s；
- 代码定位到 `wan_video_text_encoder_qwen3_5.py` 的 `_get_qwen_prompt_embeds` 和 `aceworld_video_3_1_pre_wam.py` 的 `WanVideoUnit_PromptEmbedder`；
- 确认 tokenizer/processor 的输出只依赖样本本身（prompt + 首帧图像），不依赖当前 step 的 NPU 中间结果，因此可以提前。

**方案（提前）**：
- 新增 `qwen_prompt_preprocess.py`：`get_qwen_processor` / `build_qwen_model_inputs` / `tensor_model_inputs` / `attach_worker_qwen_model_inputs`；
- `multi_sample_collate_fn` 在 DataLoader worker 内执行 chat template + processor，输出 `input_ids` / `attention_mask` / `pixel_values` / `image_grid_thw`；
- `PromptEmbedder` 接收 `qwen_model_inputs`，有预计算结果时跳过 tokenizer/processor；
- DataLoader 参数配套：`prefetch_factor` 2→4、`persistent_workers=True`、`pin_memory=True`、`QW_QWEN_WORKER_THREADS=1`；
- 预计算失败时打印日志并回退到模型内 processor 路径，不中断训练。

**结果**：
- `tokenizer/processor` **0.328 → 0.159 s**（−51.5%）。
- `preprocess` 小计 0.852 → 0.680 s（−20.2%），`preprocess` 总计 1.932 → 1.742 s。
- stage `speed` 5.29 → 4.851 s（−0.44 s）。
- 前 3 步 loss 与 baseline 完全一致（175.2460 / 206.7786 / 84.6423）。

**注意事项**：
- 本轮 `prefetch_factor` / `persistent_workers` / `pin_memory` 与预计算一起验证，没有单独 A/B，**不能把全部 0.44 s 归因于 processor 预计算**；严格归因需要隔离实验。
- 预计算只覆盖 processor/tokenizer + 首帧图像，`text_lm`（Qwen 模型前向）不变。

**经验**：
- 判断“能否提前”的关键：工作输入是否只依赖样本，不依赖当前 step 的模型输出。
- 下沉到 worker 后必须配套预取参数，否则 worker 处理时间可能仍暴露到 step 上。
- worker 内要控制线程数，避免多 worker 时 torch 线程池超订。

---

## C3 · VAE 逐样本循环 launch gap（Kairos3.2 / Wan2.2）

**背景**：同 C1，BS16/rank，`WanVideoVAE.encode()` 默认 `batch_encode=False`，对 batch 内每个样本逐样本循环调用 VAE encoder。

**现象**：
- VAE encode 窗口 wall **1273.4 ms**，kernel 时间 **984.6 ms**，free **288.8 ms（22.7%）**。
- `Conv3DV2` 调用 **3266 次**，单次 kernel 很短，launch 间隙密集。
- 属于“小算子风暴 + 逐样本循环”型空泡，不是单个 Conv3D 算子慢。

**定位过程**：
- 从 `kernel_details.csv` 按 VAE encode 窗口统计 wall / kernel / free；
- `op_statistic.csv` 显示 `Conv3DV2` 次数 3266，但总时间没有明显下降空间；
- 代码定位 `wan_video_vae.py` 的 `encode()` 循环，`aceworld_video_3_1_pre_wam.py` 的 `WanVideoUnit_InputVideoEmbedder` / `ImageEmbedderFused` 未传 `batch_encode`。

**方案（加速 / batch）**：
- `WanVideoVAE.encode(batch_encode=True)` 支持 `[B,C,T,H,W]` tensor 和 `[C,1,H,W]` tensor list；
- encode 结束调用 `self.model.clear_cache()`，避免 batch 级 VAE temporal cache 留到 DiT forward；
- `InputVideoEmbedder` / `ImageEmbedderFused` 读取 `QWORLD_VAE_BATCH_ENCODE` 并传入，默认 0 保持历史行为。

**结果**：
- VAE encode wall **1273.4 → 835.1 ms**（−34.4%），kernel 984.6 → 788.3 ms（−19.9%），free **288.8 → 46.8 ms（−83.8%）**。
- `Conv3DV2` 次数 **3266 → 206**（−93.7%）；总设备时间略升（871.4 → 929.1 ms），说明收益主要来自减少 launch/搬运，不是提高 Conv3D 吞吐。
- stage `preprocess` 2.330 → 2.009 s（−13.8%），VAE 合计 1.334 → 1.028 s（−22.9%），整步 `speed` 6.041 → 5.758 s（−4.7%）。
- loss 最大相对差约 0.014%，属于数值舍入/运行噪声。
- 代价：`fwd_peak` +2.59 GB（39.02 → 41.61 GB），`bwd_peak` 不变。

**经验**：
- “free 高但 kernel 时间没降”是 launch gap 的典型特征；优先 batch / 融合 / 减少调用，而不是继续优化单 kernel。
- batch 化会提高显存峰值，必须检查 HBM 余量。
- 临时 cache 要及时清理，避免污染后续阶段。

---

## C4 · 伪通信：barrier / dataloader rank skew（A800 vs 910B 对比）

**背景**：8×910B3 BS16 vs 8×A800 BS16，训练场景，DeepSpeed ZeRO-1。

**现象**：
- 910B 的 `Communication` 中约 **652 ms/step** 是 `QW_ITER_BARRIER`（step 3 为 313 ms、step 4 为 991 ms），看起来像通信瓶颈。
- A800 的 per-iteration barrier 只有约 **145 ms/step**。
- 但 910B 扣掉 barrier 后实际梯度通信约 **113 ms/step**，仅为 A800 的 1/8。

**归因**：
- 这不是通信本身慢，而是 **dataloader rank skew**：各 rank 进入 barrier 的时间不一致，先到的 rank 在 barrier 上等最慢的 rank。
- `npu-smi` / timeline 显示设备利用率周期性掉零；`data` 阶段和 worker 处理时间在各 rank 间不一致。
- 如果只看 `Communication(Not Overlapped)` 或“有 barrier”，会误判成通信问题。

**方案（消除 / 均衡）**：
- 先拉平 dataloader：`num_workers`、`prefetch_factor`、`persistent_workers`、`pin_memory`、worker 线程数；
- 检查样本长度分布、视频解码耗时、数据分片是否均匀；
- 若 `QW_ITER_BARRIER` 只是为计时/保险，评估删除；删除前确认无正确性依赖；
- 注意：即使删掉 barrier，等待也可能转移到梯度 allreduce，需要用最终 E2E 验证。

**经验**：
- **所有 rank 同时等 ≠ 通信慢**。先看各 rank 的 `Computing` 和到达 collective 的时间。
- `dist.barrier()` 会把数据加载不均衡放大成“通信时间”。
- 多卡场景默认只采 rank0 时，无法判断 rank skew；需要补采全 rank 或至少看各 rank 日志。

---

## C5 · 通信优化归因陷阱：contiguous_gradients vs overlap_comm

**背景**：同 C1，BS16 训练，DeepSpeed ZeRO-1 反向优化。

**现象**：
- `Communication(Not Overlapped)` 约占 17.6%，反向 `backward` 2.445 s。
- 初始候选：`overlap_comm=true`、`contiguous_gradients=false`、bucket 100 MB。
- 结果 `backward` 2.445 → 2.250 s（−8.0%），`Communication(Not Overlapped)` 1.074 → 0.766 s（−28.7%）。

**隔离实验**：
- `overlap_comm=true + contiguous_gradients=true`（50/200/500 MB）反而更慢（compute 约 2.03 s）。
- `overlap_comm=false + contiguous_gradients=false + bucket 100 MB`：`compute` 1.938 → 1.779 s，`backward` 2.530 s。
- `overlap_comm=true + contiguous_gradients=false + bucket 100 MB`：`compute` 1.765 s，比不 overlap 只多约 1%。
- 结论：**主要收益来自关闭 `contiguous_gradients`，不是 `overlap_comm`**。

**经验**：
- 通信优化经常把多个参数打包成“一个候选”，但必须做隔离实验，否则归因错误。
- `contiguous_gradients=true` 会带来梯度 contiguous 拷贝开销，虽然能降低峰值显存（−7~8 GB），但不等于速度收益。
- `overlap_comm` 的收益可能很小（约 1%），不要把它当作主要收益来源。
- 优化后的 `Free` 可能上升（本例 Free 0.500 → 0.630 s），因为通信等待减少后，其他 host 空泡相对显现；必须看总时间，不要只看 Free。

---

## 案例复用检查清单

遇到新空泡时，先问：

- [ ] 空泡指纹（前后算子）是否与某个案例相似？
- [ ] gap 是否在阶段边界上（案例 C1/C2）还是阶段内部（C3）？
- [ ] 是否所有 rank 同时等（C4），还是单 rank 空泡？
- [ ] 改多个参数时，是否能隔离出真正有效的变量（C5）？
- [ ] 案例的模型结构、dtype、CANN/torch_npu 版本是否与当前环境接近？
- [ ] 案例收益是否依赖特定 batch/分辨率/worker 配置？
