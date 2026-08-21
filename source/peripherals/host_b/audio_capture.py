"""真实声音设备访问，只取得原始32位浮点取样。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioDevice:
    index: int
    name: str
    input_channels: int
    output_channels: int
    sample_rate: int
    loopback: bool


def _audio_module():
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise RuntimeError("声音设备访问需要安装 PyAudioWPatch") from exc
    return pyaudio


def _devices_from_manager(manager) -> tuple[AudioDevice, ...]:
    loopback_indices = {
        int(info["index"]) for info in manager.get_loopback_device_info_generator()
    }
    devices: list[AudioDevice] = []
    for index in range(manager.get_device_count()):
        info = manager.get_device_info_by_index(index)
        devices.append(
            AudioDevice(
                index=index,
                name=str(info.get("name", "")),
                input_channels=int(info.get("maxInputChannels", 0)),
                output_channels=int(info.get("maxOutputChannels", 0)),
                sample_rate=int(round(float(info.get("defaultSampleRate", 0)))),
                loopback=index in loopback_indices,
            )
        )
    return tuple(devices)


def list_audio_devices() -> tuple[AudioDevice, ...]:
    pyaudio = _audio_module()
    manager = pyaudio.PyAudio()
    try:
        return _devices_from_manager(manager)
    finally:
        manager.terminate()


def select_audio_device(
    devices: tuple[AudioDevice, ...],
    exact_device_name: str,
    *,
    channels: int,
    require_loopback: bool,
) -> AudioDevice:
    """按器官所需的客观设备属性确定唯一声音端点。"""

    matches = tuple(
        device
        for device in devices
        if device.name.casefold() == exact_device_name.casefold()
        and device.input_channels >= channels
        and device.sample_rate == 48_000
        and device.loopback is bool(require_loopback)
    )
    if len(matches) != 1:
        role = "电脑声音回放" if require_loopback else "普通声音输入"
        raise RuntimeError(
            f"名称为 {exact_device_name!r}、48,000次采集的{role}设备数量不是1"
        )
    return matches[0]


class Float32AudioCapture:
    """从一个确定设备持续取得48,000次采集的原始声音。"""

    def __init__(
        self,
        exact_device_name: str,
        *,
        channels: int,
        require_loopback: bool,
        frames_per_block: int = 192,
    ) -> None:
        if not exact_device_name.strip():
            raise ValueError("必须填写完整声音设备名称")
        if channels not in (1, 2):
            raise ValueError("声音采集只接受一声道或两声道")
        if frames_per_block <= 0:
            raise ValueError("每块采样数量必须大于零")
        pyaudio = _audio_module()
        self._module = pyaudio
        self._manager = pyaudio.PyAudio()
        try:
            selected = select_audio_device(
                _devices_from_manager(self._manager),
                exact_device_name,
                channels=channels,
                require_loopback=require_loopback,
            )
            info = self._manager.get_device_info_by_index(selected.index)
        except BaseException:
            self._manager.terminate()
            raise
        self.channels = int(channels)
        self.sample_rate = 48_000
        self.frames_per_block = int(frames_per_block)
        try:
            self._stream = self._manager.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=selected.index,
                frames_per_buffer=self.frames_per_block,
            )
        except BaseException:
            self._manager.terminate()
            raise

    def read(self) -> bytes:
        return self._stream.read(self.frames_per_block, exception_on_overflow=False)

    def close(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._manager.terminate()


RawAudioProbe = Float32AudioCapture
