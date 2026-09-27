"""Publish a completed local artifact directory without replacing evidence."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path


def publish_directory(stage: Path, destination: Path) -> None:
    """Use Linux renameat2 so another immutable directory cannot be replaced."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "atomic directory publication unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
