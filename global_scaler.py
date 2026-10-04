#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, List
#基础数据结构
AVAILABLE_GPU_INDICES = [0, 1, 2, 3]
YAML_TEMPLATE = "deepseek-r1-dstill-qwen-1.5b-gpu-{gpu}.yaml"


@dataclass
class InstanceCounts:
    """当前各类型实例数量统计"""
    interactive: int = 0
    mixed: int = 0
    batch: int = 0


@dataclass
class ClusterInstanceState:
    """
    记录GPU与实例类型的映射关系
    - gpu_role[gpu_index] = "interactive" / "mixed" / "batch" / None
    """
    gpu_role: Dict[int, Optional[str]] = field(default_factory=lambda: {g: None for g in AVAILABLE_GPU_INDICES})

    def count_instances(self) -> InstanceCounts:
        counts = InstanceCounts()
        for role in self.gpu_role.values():
            if role == "interactive":
                counts.interactive += 1
            elif role == "mixed":
                counts.mixed += 1
            elif role == "batch":
                counts.batch += 1
        return counts


@dataclass
class GlobalQueueStats:
    """
    全局队列与负载统计的占位结构。
    只定义定义需要的输入，具体指标需要用排队论相关思想计算得到

    - num_interactive_running_instances:
        当前真正有交互式请求在跑的实例数（交互式 + 混合实例中正在忙的实例数）
    - num_interactive_instances_total:
        交互式实例总数
    - num_mixed_instances_total:
        混合实例总数
    - num_batch_groups_waiting_long:
        等待时间 > TTFT_SLO 的批处理请求组数
    - num_batch_groups_total:
        当前批处理请求总组数
    - mixed_has_idle_capacity:
        混合实例是否有明显闲置资源（True 表示“还有富余，就尽量别扩批处理实例”）
    - batch_instances_have_active_requests:
        当前是否有批处理实例正在处理批处理请求（True 则说明不能随便缩容）
    """
    num_interactive_running_instances: int
    num_interactive_instances_total: int
    num_mixed_instances_total: int

    num_batch_groups_waiting_long: int
    num_batch_groups_total: int

    mixed_has_idle_capacity: bool
    batch_instances_have_active_requests: bool

# 实例创建/删除封装
class GlobalInstanceManager:
    """
    通过 kubectl apply/delete 管理 vLLM 模型实例。

    - 把“一个 GPU 对应一个 YAML 文件”映射为一个实例
    """

    def __init__(self, yaml_template: str = YAML_TEMPLATE):
        self.yaml_template = yaml_template
        self.state = ClusterInstanceState()

    def _run_kubectl(self, args: list):
        """封装 kubectl 命令调用。"""
        cmd = ["kubectl"] + args
        print(f"[GlobalInstanceManager] 执行命令: {' '.join(cmd)}", file=sys.stderr)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[GlobalInstanceManager] kubectl 命令失败: {e}", file=sys.stderr)

    def create_instance(self, role: str) -> Optional[int]:
        """
        创建一个给定类型的实例：
        - 找到第一个空闲 GPU
        - 对应 YAML 执行 kubectl apply
        - 在状态里标记为该角色
        """
        for gpu, current_role in self.state.gpu_role.items():
            if current_role is None:
                yaml_path = self.yaml_template.format(gpu=gpu)
                print(f"[GlobalInstanceManager] 创建 {role} 实例: GPU={gpu}, YAML={yaml_path}", file=sys.stderr)
                self._run_kubectl(["apply", "-f", yaml_path])
                self.state.gpu_role[gpu] = role
                return gpu

        print("[GlobalInstanceManager] 没有空闲 GPU，无法创建新的实例", file=sys.stderr)
        return None

    def delete_instance(self, role: str) -> Optional[int]:
        """
        删除一个给定类型的实例：
        - 找到一个该类型的 GPU
        - 对应 YAML 执行 kubectl delete
        - 在状态里清空该 GPU 映射
        """
        # 尽量“后扩先缩”，避免频繁打扰低号GPU
        for gpu in sorted(self.state.gpu_role.keys(), reverse=True):
            if self.state.gpu_role[gpu] == role:
                yaml_path = self.yaml_template.format(gpu=gpu)
                print(f"[GlobalInstanceManager] 删除 {role} 实例: GPU={gpu}, YAML={yaml_path}", file=sys.stderr)
                self._run_kubectl(["delete", "-f", yaml_path])
                self.state.gpu_role[gpu] = None
                return gpu

        print(f"[GlobalInstanceManager] 当前无 {role} 类型实例可删除", file=sys.stderr)
        return None

    def get_counts(self) -> InstanceCounts:
        return self.state.count_instances()


# IBP/BBP计算与全局伸缩策略
def compute_ibp(stats: GlobalQueueStats) -> float:
    """
    IBP = 当前运行交互式请求的实例数 / (交互式实例总数 + 混合实例总数)
    """
    denom = stats.num_interactive_instances_total + stats.num_mixed_instances_total
    if denom <= 0:
        # 没有任何交互式 / 混合实例，则认为 IBP = 0（系统处于空载/异常态）
        return 0.0
    return stats.num_interactive_running_instances / denom


def compute_bbp(stats: GlobalQueueStats) -> float:
    """
    BBP = 等待时间 > TTFT SLO 的批处理请求组数/批处理请求总组数

    等待时间本身需要你在队列层面跟踪：
    - 每个批处理请求组入队时间
    - 当前时间 - 入队时间 > TTFT_SLO 即认为“等待时间超标”
    这里只用 num_batch_groups_waiting_long/num_batch_groups_total 来表示。
    """
    if stats.num_batch_groups_total <= 0:
        return 0.0
    return stats.num_batch_groups_waiting_long / stats.num_batch_groups_total


@dataclass
class GlobalAutoscalerConfig:
    """
    全局自动伸缩参数配置。
    """
    alpha: float = 0.7       # IBP期望交互式/混合实例利用率
    theta: float = 0.1       # 容忍区间半径

    ibp_high_hold_ticks: int = 3   # IBP > alpha+theta 连续多少个 tick 才触发扩容
    ibp_low_hold_ticks: int = 5    # IBP < alpha-theta 连续多少个 tick 才触发缩容

    bbp_high_threshold: float = 0.5   # BBP 高于该值触发批处理扩容（且混合实例无闲置）
    bbp_low_threshold: float = 0.1    # BBP 低于该值且无活跃批处理任务时，可缩容批处理实例

    tick_interval_seconds: int = 10   # 全局伸缩器主循环间隔（秒）


class GlobalAutoscaler:
    """
    全局自动伸缩决策器（模型实例层面）。
    """
    def __init__(self, instance_manager: GlobalInstanceManager, config: GlobalAutoscalerConfig):
        self.instance_manager = instance_manager
        self.config = config

        # 计数：连续“高 IBP / 低 IBP”的 tick 数
        self._ibp_high_streak = 0
        self._ibp_low_streak = 0

    def _scale_interactive_and_mixed(self, ibp: float):
        """
        根据 IBP 和阈值 α, θ 决定是否扩/缩交互式/混合实例。
        """
        alpha = self.config.alpha
        theta = self.config.theta

        counts = self.instance_manager.get_counts()

        upper = alpha + theta
        lower = alpha - theta

        # 高压：IBP > alpha+theta，可能交互式实例不足 -> 扩混合实例
        if ibp > upper:
            self._ibp_high_streak += 1
            self._ibp_low_streak = 0
            print(f"[GlobalAutoscaler] IBP={ibp:.3f} > {upper:.3f}, 高压持续 {self._ibp_high_streak} tick", file=sys.stderr)

            if self._ibp_high_streak >= self.config.ibp_high_hold_ticks:
                print("[GlobalAutoscaler] 触发扩容：优先新增一个 mixed 实例", file=sys.stderr)
                self.instance_manager.create_instance("mixed")
                self._ibp_high_streak = 0

        # 低压：IBP < alpha-theta，交互式资源明显富余 -> 优先减交互式实例
        elif ibp < lower:
            self._ibp_low_streak += 1
            self._ibp_high_streak = 0
            print(f"[GlobalAutoscaler] IBP={ibp:.3f} < {lower:.3f}, 低压持续 {self._ibp_low_streak} tick", file=sys.stderr)

            if self._ibp_low_streak >= self.config.ibp_low_hold_ticks:
                if counts.interactive > 0:
                    print("[GlobalAutoscaler] 触发缩容：优先删除一个 interactive 实例", file=sys.stderr)
                    self.instance_manager.delete_instance("interactive")
                else:
                    print("[GlobalAutoscaler] 没有 interactive 实例可缩容，跳过", file=sys.stderr)
                self._ibp_low_streak = 0

        else:
            # 在 [alpha-theta, alpha+theta] 区间内，不做伸缩，清空 streak
            print(f"[GlobalAutoscaler] IBP={ibp:.3f} 在稳定区间 [{lower:.3f},{upper:.3f}] 内，不做伸缩", file=sys.stderr)
            self._ibp_high_streak = 0
            self._ibp_low_streak = 0

    def _scale_batch(self, bbp: float, stats: GlobalQueueStats):
        """
        根据 BBP 决定是否扩/缩批处理实例。
        """
        counts = self.instance_manager.get_counts()
        cfg = self.config

        # 扩容条件：
        # 1) BBP > 高阈值 -> 批处理请求中“很多组等待时间超过 TTFT SLO”，说明积压严重
        # 2) mixed_has_idle_capacity == False -> 混合实例已经基本忙满，不能再指望混合帮忙
        if bbp > cfg.bbp_high_threshold and not stats.mixed_has_idle_capacity:
            print(
                f"[GlobalAutoscaler] BBP={bbp:.3f} > {cfg.bbp_high_threshold:.3f} "
                f"且混合实例无明显闲置，触发批处理扩容",
                file=sys.stderr
            )
            self.instance_manager.create_instance("batch")

        # 缩容条件：
        # 1) BBP < 低阈值 -> 大多数批处理请求组等待时间都在 SLO 内
        # 2) 没有活跃批处理任务 -> 防止正在跑的任务被“掐掉”
        elif bbp < cfg.bbp_low_threshold and not stats.batch_instances_have_active_requests:
            if counts.batch > 0:
                print(
                    f"[GlobalAutoscaler] BBP={bbp:.3f} < {cfg.bbp_low_threshold:.3f} "
                    f"且无活跃批处理任务，触发批处理缩容",
                    file=sys.stderr
                )
                self.instance_manager.delete_instance("batch")
            else:
                print("[GlobalAutoscaler] 当前无 batch 实例可缩容", file=sys.stderr)
        else:
            print(
                f"[GlobalAutoscaler] BBP={bbp:.3f} 未触发批处理扩/缩容条件 "
                f"(mixed_idle={stats.mixed_has_idle_capacity}, "
                f"batch_active={stats.batch_instances_have_active_requests})",
                file=sys.stderr
            )

    def one_tick(self, stats: GlobalQueueStats):
        """
        每个 tick 调用一次，读取当前全局队列 / 实例状态，执行一次伸缩决策。
        """
        ibp = compute_ibp(stats)
        bbp = compute_bbp(stats)

        counts = self.instance_manager.get_counts()
        print(
            f"[GlobalAutoscaler] Tick: "
            f"IBP={ibp:.3f}, BBP={bbp:.3f}, "
            f"counts(interactive={counts.interactive}, "
            f"mixed={counts.mixed}, batch={counts.batch})",
            file=sys.stderr
        )

        # 1) 先处理 interactive + mixed 的伸缩
        self._scale_interactive_and_mixed(ibp)

        # 2) 再处理 batch 的伸缩
        self._scale_batch(bbp, stats)



# 获取全局队列统计

def fetch_global_queue_stats() -> GlobalQueueStats:
    """
    这里是一个占位函数：真正的系统里，你需要从以下地方获取数据：
    - Prometheus metrics
    - Kafka / Redis 队列长度与等待时间
    - 自己维护的调度器状态

    下面只是一个“示意”实现，默认所有指标都是 0 / False。
    请在接入你自己的系统时，替换这里的逻辑。
    """
    # TODO: 替换为真实的队列统计逻辑
    return GlobalQueueStats(
        num_interactive_running_instances=0,
        num_interactive_instances_total=0,
        num_mixed_instances_total=0,

        num_batch_groups_waiting_long=0,
        num_batch_groups_total=0,

        mixed_has_idle_capacity=True,
        batch_instances_have_active_requests=False,
    )

# 主循环入口
def main():
    manager = GlobalInstanceManager()
    config = GlobalAutoscalerConfig()
    autoscaler = GlobalAutoscaler(manager, config)

    print("[GlobalAutoscaler] 启动全局自动伸缩决策器主循环", file=sys.stderr)
    print(f"[GlobalAutoscaler] tick 间隔: {config.tick_interval_seconds} 秒", file=sys.stderr)

    try:
        while True:
            stats = fetch_global_queue_stats()
            autoscaler.one_tick(stats)
            time.sleep(config.tick_interval_seconds)
    except KeyboardInterrupt:
        print("[GlobalAutoscaler] 收到 Ctrl+C 信号，退出", file=sys.stderr)


if __name__ == "__main__":
    main()