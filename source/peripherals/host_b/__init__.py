"""乙电脑源码。

乙电脑负责真实 Minecraft 画面和声音的采集侧工作，以及真实鼠标、
键盘和声音设备的执行侧工作。尚未确认的生命信号规格不会在这里用
临时规则补齐。
"""

from .config import HostBConfig, load_config

__all__ = ["HostBConfig", "load_config"]

