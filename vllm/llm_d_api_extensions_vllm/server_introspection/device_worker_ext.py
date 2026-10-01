# SPDX-License-Identifier: Apache-2.0
"""Worker extension class integrated into vLLM workers via `--worker-extension-cls`.

Installs `get_device_properties` and the worker side method the `devices`
endpoint plugin (`devices_plugin.py`) reaches over `engine_client.collective_rpc`.
This is vLLM's lever for adding a `collective_rpc` worker method from
external code with no core edit. `--worker-extension-cls` mixes this class
into the concrete `Worker` subclass's bases (see
`vllm.v1.worker.worker_base.WorkerWrapperBase`), so `self` below is the real
worker instance with `self.rank` / `self.local_rank` already set.

Enable with:
    --worker-extension-cls llm_d_api_extensions_vllm.server_introspection\
        .device_worker_ext.DeviceInfoWorkerExtension
"""

from typing import Any

from vllm.logger import init_logger

# "vllm." prefix required. vLLM's default logging config only attaches a
# handler to the "vllm" logger tree (propagate=False), so a bare __name__
# logger has no handler anywhere and silently drops every message.
logger = init_logger(f"vllm.{__name__}")

# Worker method name the `devices` plugin calls over `collective_rpc`. Shared
# so a rename breaks at import time rather than at RPC dispatch.
GET_DEVICE_PROPERTIES = "get_device_properties"


def _safe(fn: Any, *args: Any) -> Any:
    # Some platforms (e.g. CPU) don't implement every `current_platform`
    # device introspection method, Set to `None` instead of failing the
    # whole `collective_rpc` call for every rank.
    try:
        return fn(*args)
    except NotImplementedError:
        return None


class DeviceInfoWorkerExtension:
    def get_device_properties(self) -> dict:
        from vllm.platforms import current_platform

        device_id = self.local_rank
        capability = _safe(current_platform.get_device_capability, device_id)
        properties = {
            "rank": self.rank,
            "name": _safe(current_platform.get_device_name, device_id),
            "total_memory_bytes": _safe(current_platform.get_device_total_memory, device_id),
            "compute_capability": (
                {"major": capability.major, "minor": capability.minor}
                if capability is not None
                else None
            ),
            "num_compute_units": _safe(current_platform.num_compute_units, device_id),
        }
        logger.info("device_worker_ext: rank=%d name=%s", self.rank, properties["name"])
        return properties
