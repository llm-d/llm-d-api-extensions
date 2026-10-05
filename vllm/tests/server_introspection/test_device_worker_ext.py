# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `DeviceInfoWorkerExtension.get_device_properties`.

Fast with no GPU. `vllm.platforms.current_platform` is replaced by a stub so
both the populated path and the `_safe()` fallback (platforms that raise
`NotImplementedError` for device introspection) are exercised.
"""

from types import SimpleNamespace

import pytest
import vllm.platforms

from llm_d_api_extensions_vllm.server_introspection.device_worker_ext import (
    DeviceInfoWorkerExtension,
)


class _GpuPlatform:
    def __init__(self):
        self.device_ids: list[int] = []

    def get_device_capability(self, device_id):
        self.device_ids.append(device_id)
        return SimpleNamespace(major=9, minor=0)

    def get_device_name(self, device_id):
        return "H100-SXM5-80GB"

    def get_device_total_memory(self, device_id):
        return 85_899_345_920

    def num_compute_units(self, device_id):
        return 132


class _UnsupportedPlatform:
    def get_device_capability(self, device_id):
        raise NotImplementedError

    def get_device_name(self, device_id):
        raise NotImplementedError

    def get_device_total_memory(self, device_id):
        raise NotImplementedError

    def num_compute_units(self, device_id):
        raise NotImplementedError


class _BrokenNamePlatform(_GpuPlatform):
    def get_device_name(self, device_id):
        raise RuntimeError("boom")


class _Worker(DeviceInfoWorkerExtension):
    """Stands in for the vLLM `Worker` the extension is mixed into."""

    def __init__(self, rank: int, local_rank: int):
        self.rank = rank
        self.local_rank = local_rank


def _use_platform(monkeypatch, platform) -> None:
    monkeypatch.setattr(vllm.platforms, "current_platform", platform)


class TestGetDeviceProperties:
    def test_populated_platform(self, monkeypatch):
        _use_platform(monkeypatch, _GpuPlatform())
        props = _Worker(rank=3, local_rank=1).get_device_properties()
        assert props == {
            "rank": 3,
            "name": "H100-SXM5-80GB",
            "total_memory_bytes": 85_899_345_920,
            "compute_capability": {"major": 9, "minor": 0},
            "num_compute_units": 132,
        }

    def test_queries_local_rank_device(self, monkeypatch):
        platform = _GpuPlatform()
        _use_platform(monkeypatch, platform)
        _Worker(rank=5, local_rank=1).get_device_properties()
        assert platform.device_ids == [1]

    def test_unsupported_platform_falls_back_to_none(self, monkeypatch):
        _use_platform(monkeypatch, _UnsupportedPlatform())
        props = _Worker(rank=0, local_rank=0).get_device_properties()
        assert props == {
            "rank": 0,
            "name": None,
            "total_memory_bytes": None,
            "compute_capability": None,
            "num_compute_units": None,
        }

    def test_other_exceptions_propagate(self, monkeypatch):
        # `_safe()` only swallows NotImplementedError. Real failures must surface.
        _use_platform(monkeypatch, _BrokenNamePlatform())
        with pytest.raises(RuntimeError, match="boom"):
            _Worker(rank=0, local_rank=0).get_device_properties()
