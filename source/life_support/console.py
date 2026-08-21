"""控制台只暴露观察、教学和睡眠。"""

from __future__ import annotations

import json
from collections.abc import Callable

from .runtime import HostARuntime


class Console:
    def __init__(self, runtime: HostARuntime) -> None:
        self.runtime = runtime

    def observations(self):
        return self.runtime.observation.read_all()

    def start_teaching(self) -> None:
        self.runtime.start_teaching()

    def stop_teaching(self) -> None:
        self.runtime.stop_teaching()

    def sleep(self) -> None:
        self.runtime.sleep()


def run_console(
    console: Console,
    *,
    status: Callable[[], dict[str, object]],
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    """运行不接触生命信号的中文控制台。"""

    write("控制台已连接。命令：状态、观察 [数量]、开始教学、停止教学、睡眠")
    while True:
        command = read("生命> ").strip()
        if command == "状态":
            write(json.dumps(status(), ensure_ascii=False, indent=2))
        elif command.startswith("观察"):
            parts = command.split()
            count = 10 if len(parts) == 1 else max(1, int(parts[1]))
            write(json.dumps(console.observations()[-count:], ensure_ascii=False, indent=2))
        elif command == "开始教学":
            console.start_teaching()
            write("教学已开始；Brain自己的动作倾向仍保存。")
        elif command == "停止教学":
            console.stop_teaching()
            write("教学已停止。")
        elif command == "睡眠":
            console.sleep()
            write("路径已按出生固定量削弱，个体已储存并关机。")
            return
        elif command:
            write("未知命令；只能观察、开始/停止教学或睡眠。")
