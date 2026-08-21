"""乙电脑程序使用的明确错误类型。"""


class HostBError(RuntimeError):
    """乙电脑程序基础错误。"""


class ConfigurationError(HostBError):
    """配置文件不完整或自相矛盾。"""


class UnconfirmedSpecificationError(HostBError):
    """请求进入了尚未由用户确认的运行规则。"""


class ProtocolError(HostBError):
    """局域网传递帧无效。"""

