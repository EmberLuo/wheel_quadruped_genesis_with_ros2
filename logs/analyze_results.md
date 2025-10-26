# analyze_results

## tensorboard看训练结果

```shell
tensorboard --logdir="logs/wheel-quadruped-walking"
```

打开浏览器，输入<http://localhost:6006/>即可查看训练结果。

## 训练指标

1. 稳定性

* **NaN/Inf**：loss、grad、obs、reward 是否出现 NaN/Inf（一旦有，先排查再训练）。
* **学习率/梯度范数**：是否爆高或长期为 0（梯度裁剪是否生效）。
* **Episode length / Done 原因**：是否一开始就疯狂 “早终止/摔倒”。

1. PPO/Actor-Critic 核心

* **KL divergence**（或 `approx_kl`）：建议每次更新 **0.01–0.02** 左右；持续 >0.05 多半过大步、会抖/不稳。
* **clip fraction**：落在 **0.1–0.3** 较理想；长期接近 0 代表学习停滞，过高说明更新太激进。
* **entropy**：不要太低（避免过早收敛）；若骤降，增大探索或调高词汇/动作噪声。

1. Value 相关

* **value loss**：应逐步下降并稳定；长时间高位或震荡 → 试着增大 value loss 系数、增大 batch、调 GAE(λ)。
* **预测回报 vs. 真实回报**：两者差距是否缩小（可以看 “Explained variance” > **0.6** 更好）。

1. 任务级 KPI

* **成功率 / 失败率（跌倒数）/ 平均回合时长**：是否单调向好；如果课程加难后快速恶化，先稳住难度。
* **主奖励曲线（总回报）**：单调上升并逐渐平滑；突刺+回落通常是更新过猛或课程推进过快。

1. 行为与物理质量

* **动作统计**：`action_mean/std`、力矩/速度饱和比例；长期饱和 → action_scale 太大或奖励不平衡。
* **对称/平滑/接触质量**：如 `action_mirror`、`dof_acc`、`feet_contact/stumble` 等是否持续改善。
* **碰撞/越界/安全惩罚**：是否逐步接近 0。

1. 课程/难度进度

* **curriculum 进度**（速度/地形/扰动范围）：提到顶后，核心质量指标还能不能继续变好；不行就先“冻结难度”。

## 快速阈值参考（PPO 常用）

| 指标                   | 推荐起始范围/期望趋势           |
| -------------------- | --------------------- |
| KL/更新                | 0.01–0.02（>0.05 通常过猛） |
| clip fraction        | 0.1–0.3（≈0 停滞，过高太激进）  |
| entropy              | 训练前中期保持不低，后期缓降        |
| value loss           | 逐步下降并稳住               |
| explained variance   | >0.6 更好               |
| episode length / 成功率 | 随训练稳步上升               |
