# SPDX-License-Identifier: Apache-2.0
"""Contract tests: does the installed vLLM still have what the plugins read?

The plugins degrade instead of crashing when vLLM renames or removes
something (see `common.py`) which also means nothing fails loudly when that
happens. These tests are the loud part. They run the real plugin code against
stand ins that have exactly the attributes of the installed vLLM's config
classes and fail on any drift the plugins report, naming what changed.

The other unit tests cannot do this, their config objects are hand built and
so always match what the plugin expects. No engine, GPU or model is needed,
only an importable `vllm`. CI runs this against both the released and the
nightly vLLM.
"""

import asyncio
import dataclasses
import inspect
from argparse import Namespace
from typing import Any

import pytest
import vllm.platforms
from fastapi import FastAPI
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from llm_d_api_extensions_vllm.server_introspection import common
from llm_d_api_extensions_vllm.server_introspection.common import ENGINE_TASKS, drift_events
from llm_d_api_extensions_vllm.server_introspection.config_plugin import ServerConfigPlugin
from llm_d_api_extensions_vllm.server_introspection.device_worker_ext import (
    DeviceInfoWorkerExtension,
)
from llm_d_api_extensions_vllm.server_introspection.devices_plugin import ServerDevicesPlugin
from llm_d_api_extensions_vllm.server_introspection.kv_cache_plugin import (
    _KIND_TO_MODEL,
    ServerKVCachePlugin,
    _capacity_from_vllm_config,
)

_PLUGINS = (ServerConfigPlugin, ServerDevicesPlugin, ServerKVCachePlugin)

_DRIFT_HELP = (
    "The installed vLLM no longer matches what the plugins read. Update the "
    "plugin (and schema/README) for each of"
)


def _assert_no_drift() -> None:
    events = drift_events()
    assert not events, f"{_DRIFT_HELP}: {events}"


def _field_default(field: dataclasses.Field) -> Any:
    """The default vLLM declares for a config field, `None` if it has none."""
    default: Any = field.default
    factory: Any = field.default_factory
    if isinstance(default, FieldInfo):  # pydantic `Field(...)` as the default
        default, factory = default.default, default.default_factory
    if default not in (dataclasses.MISSING, PydanticUndefined):
        return default
    if factory not in (dataclasses.MISSING, None):
        try:
            return factory()
        except Exception:  # e.g. needs a GPU platform
            return None
    return None


def _shaped_like(vllm_cls: type, **overrides: Any) -> Any:
    """An object with exactly the public attributes of vLLM class `vllm_cls`.

    Fields carry vLLM's own defaults so their types are checked against the
    response schema too. Anything else raises `AttributeError`, as it would
    on a real instance. Named after `vllm_cls` so drift reads `ModelConfig.x`.
    """
    values = {f.name: _field_default(f) for f in dataclasses.fields(vllm_cls)}
    # Properties and other non field attributes exist but have no cheap value.
    values.update(dict.fromkeys(set(dir(vllm_cls)) - set(values)))
    unknown = set(overrides) - set(values)
    assert not unknown, f"{vllm_cls.__name__} has no attribute(s) {sorted(unknown)}"
    values.update(overrides)

    def __getattr__(self: Any, name: str) -> Any:
        try:
            return values[name]
        except KeyError:
            raise AttributeError(name) from None

    return type(vllm_cls.__name__, (), {"__getattr__": __getattr__})()


def _real_shaped_vllm_config(*, kv_connector: str | None = None) -> Any:
    from vllm.config import (
        CacheConfig,
        KVTransferConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    return _shaped_like(
        VllmConfig,
        model_config=_shaped_like(ModelConfig),
        cache_config=_shaped_like(CacheConfig),
        scheduler_config=_shaped_like(SchedulerConfig),
        parallel_config=_shaped_like(ParallelConfig),
        kv_transfer_config=(
            _shaped_like(KVTransferConfig, kv_connector=kv_connector) if kv_connector else None
        ),
    )


def test_import_time_dependencies_resolved():
    # `common.py` falls back when these imports fail, so check they did not.
    from vllm.tasks import GENERATION_TASKS, POOLING_TASKS

    assert common.init_logger is not None, "vllm.logger.init_logger is gone"
    assert common.GENERATION_TASKS is not None, "vllm.tasks task tuples are gone"
    assert tuple(GENERATION_TASKS) + tuple(POOLING_TASKS) == ENGINE_TASKS


def test_fallback_engine_tasks_match_vllm():
    assert common._FALLBACK_ENGINE_TASKS == ENGINE_TASKS, (
        "vLLM's task names changed, update _FALLBACK_ENGINE_TASKS in common.py"
    )


# NixlConnector also exercises the `vllm.envs` side channel settings.
@pytest.mark.parametrize("kv_connector", [None, "NixlConnector"])
def test_config_plugin_reads_only_what_vllm_config_has(kv_connector):
    state = FastAPI().state
    state.vllm_config = _real_shaped_vllm_config(kv_connector=kv_connector)

    asyncio.run(ServerConfigPlugin().init_state(None, state, Namespace(served_model_name=["m"])))

    _assert_no_drift()
    assert state.server_config_response is not None
    assert (state.server_config_response.kv_transfer is not None) == bool(kv_connector)


def test_kv_cache_plugin_reads_only_what_vllm_config_has():
    _capacity_from_vllm_config(_real_shaped_vllm_config())
    _assert_no_drift()


def test_kv_cache_group_kinds_all_have_a_schema():
    from vllm.v1.kv_cache_interface import KVCacheSpecKind

    missing = {kind.value for kind in KVCacheSpecKind} - set(_KIND_TO_MODEL)
    assert not missing, (
        f"vLLM has KV cache spec kinds with no group schema: {sorted(missing)}. "
        "They are served as UnknownGroupSpec until one is added."
    )


def test_worker_extension_reads_only_what_platform_has(monkeypatch):
    from vllm.platforms.interface import Platform

    # The base class has every device method any platform can be asked for.
    monkeypatch.setattr(vllm.platforms, "current_platform", Platform())

    class _Worker(DeviceInfoWorkerExtension):
        rank = 0
        local_rank = 0

    _Worker().get_device_properties()
    _assert_no_drift()


def test_engine_client_still_has_collective_rpc():
    from vllm.engine.protocol import EngineClient

    assert callable(getattr(EngineClient, "collective_rpc", None))


@pytest.mark.parametrize("plugin_cls", _PLUGINS)
def test_plugins_satisfy_endpoint_plugin_protocol(plugin_cls):
    from vllm.plugins.endpoint_plugins.interface import EndpointPlugin

    plugin = plugin_cls()
    assert isinstance(plugin, EndpointPlugin)
    for method in ("attach_router", "init_state"):
        expected = list(inspect.signature(getattr(EndpointPlugin, method)).parameters)
        actual = list(inspect.signature(getattr(plugin_cls, method)).parameters)
        assert actual == expected, f"EndpointPlugin.{method} signature changed"


@pytest.mark.parametrize("plugin_cls", _PLUGINS)
def test_vllm_calls_init_state_the_way_plugins_expect(plugin_cls):
    # A raising `init_state` aborts server startup only because vLLM does not
    # catch it. If vLLM starts isolating plugin failures this still passes.
    from vllm.plugins.endpoint_plugins.interface import init_endpoint_plugins_state

    state = FastAPI().state
    state.endpoint_plugins = [plugin_cls()]
    asyncio.run(init_endpoint_plugins_state(None, state, Namespace()))
