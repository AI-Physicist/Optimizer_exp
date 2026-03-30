## 项目目标

本项目在可控条件下研究不同优化器在小型 Transformer 上的训练行为差异。

核心问题：

1. 优化器在合成任务上的表现有何差异？
2. 优化器在小规模真实文本语言建模任务上的表现有何差异？
3. 从非凸优化动力学视角看，优化器的轨迹几何和收敛机制有何差异？

---

## 实验设置

### 模型

- 小型 Decoder-only Transformer（约 10M 参数）

### 对比优化器

- AdamW
- SGD（带 momentum）
- RMSprop（及其变体）
- Adafactor

### 控制变量

- 相同模型结构
- 相同数据集（每个实验内）
- 相同 batch size
- 相同步数
- 相同初始化

### 调节变量

- 学习率（按任务和优化器分别调参）

---

## 实验一：合成任务（Synthetic）

### 任务定义

- 随机 token 序列
- 下一 token 预测：`y = roll(x)`

### 关键性质

- 真实数据分布近似均匀
- 理论下界：`loss = ln(vocab_size)`

### 观察

- 各优化器都能收敛到接近理论下界
- 优化器间差异较小
- RMSprop / SGD 在部分设置下可略优于 AdamW

### 解释

该任务简单、条件较好，大多数合理优化器都能到达近似相同最优区域。

---

## 实验二：真实文本任务（Harry Potter，约 2 万字符）

### 任务定义

- 字符级下一 token 预测

### 相比合成任务的差异

- token 分布非均匀
- 存在明显局部结构（单词、标点、短语模式）
- 梯度统计更复杂

### 结果（1000 steps，调好学习率，3 seeds）

| optimizer | final_loss_mean | final_loss_std | avg_step_time |
|---|---:|---:|---:|
| **AdamW** | **1.862** | 0.029 | 0.0286 |
| Adafactor | 2.354 | 0.039 | 0.0462 |
| SGD | 2.434 | 0.024 | **0.0272** |
| RMSprop | 2.560 | 0.058 | 0.0284 |

说明：

- 单 seed 下在 300-step 与 1000-step 之间出现的排名波动，在多 seed 平均后明显减弱。
- 该真实文本设置下的稳定排名为：**AdamW > Adafactor > SGD > RMSprop**。

---

## 实验三：非凸优化动力学分析（首轮已完成）

### 分析目标

- 不仅比较“最终 loss”，还比较训练轨迹在非凸损失面的动力学行为。

### 记录指标

- `loss`
- `grad_norm_l2`
- `update_norm_l2`
- `update_ratio = update_norm_l2 / param_norm_l2`
- `grad_update_cos`（更新方向与原始梯度方向的一致性）
- `hessian_top_eig`（Hessian 最大特征值的近似）

### 首轮结果（单 seed，1000 steps，log_every=10）

| optimizer | loss@10 | loss@100 | final_loss | grad_norm: first->last | update_ratio(均值) | grad_update_cos(均值) | hessian_top_eig: first->last |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 75.371 | 2.505 | **0.110** | 27.42 -> 0.85 | 4.19e-4 | -0.192 | 391.16 -> 33.23 |
| SGD (momentum=0.9) | 84.859 | 3.018 | 2.471 | 25.31 -> 0.28 | 1.05e-3 | -0.032 | 66.70 -> 18.98 |
| Adafactor | 76.799 | 8.865 | 2.392 | 36.54 -> 0.65 | 2.10e-2 | -0.093 | -290.03 -> 885.70 |

### 动力学解读

- **AdamW**：前期快速降 loss，梯度范数快速压低；后期进入慢速改进。`grad_update_cos` 为中等负相关，体现出自适应预条件后的更新方向不等同于原始梯度方向。`hessian_top_eig` 整体下降，轨迹从高曲率区走向更平坦区。
- **SGD（含动量）**：`grad_update_cos` 远离 `-1`（均值约 `-0.032`）并非实现错误，而是因为更新方向由“历史动量 + 当前梯度 + weight decay”共同决定，而不是纯 `-g_t`；因此与“当前 batch 梯度”不必高度对齐。
- **Adafactor**：步长比（`update_ratio`）明显更大，曲率估计波动也更强（含正负大幅变化），对应其训练过程更“激进”和不稳定的几何轨迹；最终 loss 仍明显高于 AdamW。

### 图表输出

- 单优化器六联图：
`results/real_text/dynamics/dynamics_sgd.svg`
`results/real_text/dynamics/dynamics_adafactor.svg`
`results/real_text/dynamics/optimizer_dynamics.svg`（AdamW）
- 三优化器合并六联图：
`results/real_text/dynamics/dynamics_adamw_sgd_adafactor.svg`

---

## 主要结论

### 1. 优化器差异具有明显任务依赖性

- 合成任务中差异较小
- 真实文本任务中差异更明显

### 2. AdamW 在当前真实文本设置中表现最佳

- 最低的最终 loss 均值
- 跨 seed 收敛更稳定

### 3. RMSprop 在简单任务可具竞争力，但在真实文本上较弱

- 仅靠按坐标缩放在该任务中不足以取得最佳结果

### 4. SGD 速度快但精度劣于 AdamW

- 单步时间更优
- 最终 loss 更高

### 5. 动力学视角补充了“为什么”

- 仅看最终指标不够，需要结合梯度规模、更新几何与曲率变化理解优化器行为。
- 在该任务中，AdamW 更快进入“低梯度 + 低曲率”稳定区，并取得显著更低的最终 loss。

---

## 局限性

- 模型规模较小（约 10M）
- 数据规模较小（约 2 万字符）
- 当前多 seed 仅 3 次，统计置信度仍可提升
- 非凸动力学目前是单 seed 结果，仍需多 seed 统计验证

---

## 后续工作

- 将 seed 数从 3 提升到 5+
- 延长训练步数，观察后期动力学
- 扩展到更大模型与更长序列
- 引入更真实的数据集
- 扩展动力学实验到多 seed，并给出置信区间
