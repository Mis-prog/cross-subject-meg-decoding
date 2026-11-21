"""Lazy recordings safe for both forked and spawned DataLoader workers."""
import os
from collections.abc import Mapping

import numpy as np


class RecordingStore(Mapping):
    def __init__(self, path):
        self.path = str(path)
        self._cache = {}
        self._archive = None
        self._pid = None

    def _open(self):
        if self._pid != os.getpid():
            self.close()
            self._archive = np.load(self.path)
            self._pid = os.getpid()
        return self._archive

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = self._open()[key]
        return self._cache[key]

    def __iter__(self):
        return iter(self._open().files)

    def __len__(self):
        return len(self._open().files)

    def close(self):
        if self._archive is not None:
            self._archive.close()
        self._archive = None
        self._pid = None

    def __getstate__(self):
        # Each spawned worker opens its own archive and fills its own cache.
        return {'path': self.path, '_cache': {}, '_archive': None, '_pid': None}
