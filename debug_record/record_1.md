# 🛠️ Genesis 仿真环境调试记录-Day1

本记录总结了在使用 `wheel_quadruped_genesis` 项目中遇到的两类常见错误及排查解决过程：

---

## 🐞 错误 1：`Link not found for name: trunk`

### ❓ 问题描述-固定关节折叠问题

运行时报错：

```shell
genesis.GenesisException: Link not found for name: trunk.
```

### 📌 原因分析-固定关节折叠问题

Genesis 默认会折叠固定关节（如 floating_base），将子链接（`trunk`）合并为父链接（`base`），导致 `trunk` 链接在 URDF 中不再存在。

### ✅ 解决方案-固定关节折叠问题

修改训练脚本中的连接平面链接配置，将 `trunk` 替换为 `base`：

```python
"connect_plane_links": ["base", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
```

---

## 🐞 错误 2：`Sizes of tensors must match`

### ❓ 问题描述-维度不匹配问题

运行时在 `self.obs_buf = torch.cat(...)` 时报错：

```shell
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 28 but got size 50 for tensor number 1 in the list.
```

### 📌 原因分析-维度不匹配问题

配置中的 `num_slice_obs=28` 与实际生成的 `slice_obs_buf=50` 不一致，导致观测历史拼接时报维度错误。

实际的 `slice_obs_buf` 由以下特征组成：

* base\_ang\_vel: 3
* projected\_gravity: 3
* dof\_pos deviation: 12
* dof\_vel: 16
* actions: 16
* **合计：50**

### ✅ 解决方案-维度不匹配问题

修改 `obs_cfg` 配置参数如下：

```python
obs_cfg = {
    "num_slice_obs": 50,
    "history_length": 9,
    "num_obs": 50 * (9 + 1) + 6,  # 6 是 num_commands
    ...
}
```

确保 `num_slice_obs` 和 `num_obs` 与实际观测维度保持一致。

---

## 🧪 实用指令记录

更新后重新运行指令：

```bash
conda run --live-stream --name genesis python /home/ember/GitHub/wheel_quadruped_genesis/locomotion/wheel_legged_train.py
```

---

## 📌 总结

* URDF 文件如有改动，注意 Genesis 的固定关节折叠行为；
* 配置项应与实际观测维度保持同步；
* 使用 `print(self.slice_obs_buf.shape)` 等手段动态调试观察维度；
* 出现 `Link not found` 先排查 URDF 是否有 collapse 行为。
