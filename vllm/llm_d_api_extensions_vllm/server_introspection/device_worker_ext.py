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

from .common import get_logger, read_attr

logger = get_logger(__name__)

# Worker method name the `devices` plugin calls over `collective_rpc`. Shared
# so a rename breaks at import time rather than at RPC dispatch.
GET_DEVICE_PROPERTIES = "get_device_properties"


def _safe(platform: Any, method: str, *args: Any) -> Any:
    # Some platforms (e.g. CPU) don't implement every `current_platform`
    # device introspection method, and vLLM may rename or drop one. Set to
    # `None` instead of failing the whole `collective_rpc` call for every rank.
    fn = read_attr(platform, method)
    if fn is None:
        return None
    try:
        return fn(*args)
    except NotImplementedError:
        return None


class DeviceInfoWorkerExtension:
    def get_device_properties(self) -> dict:
        from vllm.platforms import current_platform

        rank = read_attr(self, "rank")
        device_id = read_attr(self, "local_rank", 0)
        capability = _safe(current_platform, "get_device_capability", device_id)
        properties = {
            "rank": rank,
            "name": _safe(current_platform, "get_device_name", device_id),
            "total_memory_bytes": _safe(current_platform, "get_device_total_memory", device_id),
            "compute_capability": (
                {
                    "major": read_attr(capability, "major"),
                    "minor": read_attr(capability, "minor"),
                }
                if capability is not None
                else None
            ),
            "num_compute_units": _safe(current_platform, "num_compute_units", device_id),
        }
        logger.info("device_worker_ext: rank=%s name=%s", rank, properties["name"])
        return properties
