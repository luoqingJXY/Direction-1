"""严格按冻结顺序执行的睡眠与恢复。"""

from __future__ import annotations

from collections.abc import Callable

from .brain_boundary import BrainBoundary
from .memory_cache import MemoryCache
from .storage import LifeStorage


class SleepController:
    def __init__(
        self,
        brain: BrainBoundary,
        memory_cache: MemoryCache,
        storage: LifeStorage,
        close_components: Callable[[], None],
    ) -> None:
        self.brain = brain
        self.memory_cache = memory_cache
        self.storage = storage
        self.close_components = close_components
        self.closed = False

    def sleep(self) -> None:
        if self.closed:
            return
        self.brain.weaken_plastic_paths_for_sleep()
        self.memory_cache.stop_sending()
        self.storage.save(
            self.brain.export_life_structure(),
            self.brain.export_path_state(),
        )
        self.brain.shutdown()
        self.close_components()
        self.closed = True

    def restore_same_life(self) -> None:
        stored = self.storage.load()
        self.brain.restore_existing_life(stored.life_structure, stored.path_state)
        self.memory_cache.resume_sending()
        self.closed = False

