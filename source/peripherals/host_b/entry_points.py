"""当前已确认、且与乙电脑有关的连接边界。

这里只声明信息流从哪里进入或离开乙电脑，不描述人工DNA如何把这些边界
连接到脑。网络通道编号是工程地址，不属于生命信号。
"""

from __future__ import annotations

from enum import IntEnum


class Channel(IntEnum):
    VISUAL_RECEPTOR_UPDATES = 1
    AUDITORY_RECEPTOR_UPDATES = 2
    MOUSE_ACTIVITIES = 3
    KEYBOARD_ACTIVITIES = 4
    VIEW_CENTER_ACTIVITIES = 5
    VOICE_OUTPUT = 6
    CONTROL = 7


CHANNEL_CONFIG_NAMES: dict[Channel, str] = {
    Channel.VISUAL_RECEPTOR_UPDATES: "visual",
    Channel.AUDITORY_RECEPTOR_UPDATES: "audio",
    Channel.MOUSE_ACTIVITIES: "mouse",
    Channel.KEYBOARD_ACTIVITIES: "keyboard",
    Channel.VIEW_CENTER_ACTIVITIES: "view_center",
    Channel.VOICE_OUTPUT: "voice",
    Channel.CONTROL: "control",
}


UPSTREAM_CHANNELS = frozenset(
    {
        Channel.VISUAL_RECEPTOR_UPDATES,
        Channel.AUDITORY_RECEPTOR_UPDATES,
        # 普通期甲→乙执行Brain动作；教学期乙→甲返回人已经
        # 实际形成的同规格器官活动。通道号和方向不进入Brain。
        Channel.MOUSE_ACTIVITIES,
        Channel.KEYBOARD_ACTIVITIES,
        Channel.CONTROL,
    }
)

DOWNSTREAM_CHANNELS = frozenset(
    {
        Channel.MOUSE_ACTIVITIES,
        Channel.KEYBOARD_ACTIVITIES,
        Channel.VIEW_CENTER_ACTIVITIES,
        Channel.VOICE_OUTPUT,
        Channel.CONTROL,
    }
)
