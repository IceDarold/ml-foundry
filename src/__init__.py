"""Project-level initialisation helpers."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from typing import Iterable


def _dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _ensure_libomp_loaded() -> None:
    """Try to ensure libomp is available on macOS.

    Wheel distributions of packages like scikit-learn and LightGBM bundle
    `libomp.dylib`, but the dynamic loader does not look in those private
    directories by default. When it is missing we get hard-to-debug crashes.
    """
    if sys.platform != "darwin":
        return

    # If the system already finds libomp we are done.
    if ctypes.util.find_library("omp"):
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
    os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

    candidate_dirs: list[Path] = []
    for entry in sys.path:
        if "site-packages" not in entry:
            continue
        base = Path(entry)
        candidate_dirs.extend(
            [
                base / "sklearn" / ".dylibs",
                base / "torch" / "lib",
            ]
        )

    existing_env = [
        p for p in os.environ.get("DYLD_LIBRARY_PATH", "").split(":") if p
    ]

    new_dirs: list[str] = []
    for directory in candidate_dirs:
        if directory.is_dir():
            lib_path = directory / "libomp.dylib"
            if lib_path.exists():
                new_dirs.append(str(directory))

    if not new_dirs:
        return

    merged = _dedupe(new_dirs + existing_env)
    os.environ["DYLD_LIBRARY_PATH"] = ":".join(merged)
    fallback_existing = [
        p for p in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if p
    ]
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(_dedupe(new_dirs + fallback_existing))

    # Attempt to preload once so subsequent imports succeed.
    for directory in new_dirs:
        lib_path = Path(directory) / "libomp.dylib"
        try:
            ctypes.CDLL(str(lib_path))
            break
        except OSError:
            continue


_ensure_libomp_loaded()
