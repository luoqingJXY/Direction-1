"""已经形成实际声音波形后的设备播放端。"""

from __future__ import annotations


class Float32VoiceOutput:
    """播放声音器官最终形成的单声道波形，不解释声音活动。"""

    def __init__(self, exact_device_name: str, sample_rate: int) -> None:
        if not exact_device_name.strip():
            raise ValueError("必须填写完整声音输出设备名称")
        if sample_rate <= 0:
            raise ValueError("声音采样率必须大于零")
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise RuntimeError("声音播放需要安装 PyAudioWPatch") from exc
        self._module = pyaudio
        self._manager = pyaudio.PyAudio()
        self.sample_rate = int(sample_rate)
        matches = []
        for index in range(self._manager.get_device_count()):
            info = self._manager.get_device_info_by_index(index)
            if (
                str(info.get("name", "")).casefold() == exact_device_name.casefold()
                and int(info.get("maxOutputChannels", 0)) > 0
            ):
                matches.append(info)
        if len(matches) != 1:
            self._manager.terminate()
            raise RuntimeError(f"名称为 {exact_device_name!r} 的输出设备数量不是1")
        self._stream = self._manager.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            output_device_index=int(matches[0]["index"]),
        )

    def write(self, raw_float32_mono: bytes, sample_rate: int) -> None:
        if int(sample_rate) != self.sample_rate:
            raise ValueError("声音波形采样率与已打开设备不一致")
        if len(raw_float32_mono) % 4:
            raise ValueError("32位浮点声音字节长度必须是4的倍数")
        self._stream.write(raw_float32_mono)

    def close(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._manager.terminate()

