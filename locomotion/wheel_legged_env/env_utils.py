import torch


def gs_rand_float(lower: float, upper: float, shape, device: torch.device) -> torch.Tensor:
    """生成给定范围内的随机浮点数。"""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def adjust_scale(error: torch.Tensor,
                 lower_err: float, upper_err: float,
                 scale: torch.Tensor, scale_step: float,
                 min_range: float, range_cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据误差调整课程学习的比例 scale，并返回更新后的范围。
    """
    min_condition = error < lower_err
    max_condition = error > upper_err
    # 增减比例
    scale[min_condition] += scale_step
    scale[max_condition] -= scale_step
    scale.clamp_(min_range, 1)
    range_min, range_max = range_cfg
    return scale * range_min, scale * range_max


def get_relative_terrain_pos(base_pos: torch.Tensor,
                             terrain_height: torch.Tensor,
                             horizontal_scale: float) -> torch.Tensor:
    """
    对一批 (x,y) 坐标执行双线性插值，返回调整后的基座位置。注意只改变 z 轴。
    """
    x, y = base_pos[:, 0], base_pos[:, 1]
    fx, fy = x / horizontal_scale, y / horizontal_scale
    x0 = torch.floor(fx).int()
    x1 = torch.min(x0 + 1, torch.full_like(x0, terrain_height.shape[1] - 1))
    y0 = torch.floor(fy).int()
    y1 = torch.min(y0 + 1, torch.full_like(y0, terrain_height.shape[0] - 1))
    # 防止越界
    x0 = torch.clamp(x0, 0, terrain_height.shape[1] - 1)
    x1 = torch.clamp(x1, 0, terrain_height.shape[1] - 1)
    y0 = torch.clamp(y0, 0, terrain_height.shape[0] - 1)
    y1 = torch.clamp(y1, 0, terrain_height.shape[0] - 1)
    Q11 = terrain_height[y0, x0]
    Q21 = terrain_height[y0, x1]
    Q12 = terrain_height[y1, x0]
    Q22 = terrain_height[y1, x1]
    wx = fx - x0
    wy = fy - y0
    height = (1 - wx) * (1 - wy) * Q11 + wx * (1 - wy) * Q21 \
        + (1 - wx) * wy * Q12 + wx * wy * Q22
    base_pos[:, 2] = base_pos[:, 2] - height
    return base_pos
