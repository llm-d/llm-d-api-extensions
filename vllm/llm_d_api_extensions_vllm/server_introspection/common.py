# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the server introspection endpoint plugins."""

import logging

from fastapi import HTTPException, Request
from vllm.logger import init_logger
from vllm.tasks import GENERATION_TASKS, POOLING_TASKS

# Tasks for plugins that need an engine. Excludes the `render` frontend task
# because there is nothing to introspect on the CPU only render server.
ENGINE_TASKS: tuple[str, ...] = GENERATION_TASKS + POOLING_TASKS

_UNSET = object()


def get_logger(name: str) -> logging.Logger:
    # "vllm." prefix required. vLLM's default logging config only attaches a
    # handler to the "vllm" logger tree (propagate=False), so a bare __name__
    # logger has no handler anywhere and silently drops every message.
    return init_logger(f"vllm.{name}")


def no_engine_detail(endpoint: str) -> str:
    return f"{endpoint} requires an engine, which this server does not have"


def cached_response(request: Request, state_attr: str, unavailable_detail: str):
    """Return the response `init_state` cached on `app.state.<state_attr>`.

    Raises 500 when `init_state` never ran and 503 with `unavailable_detail`
    when it ran but cached `None` (the endpoint can't serve on this server).
    """
    response = getattr(request.app.state, state_attr, _UNSET)
    if response is _UNSET:
        raise HTTPException(
            status_code=500,
            detail=f"{state_attr.removesuffix('_response')} plugin state was never initialized",
        )
    if response is None:
        raise HTTPException(status_code=503, detail=unavailable_detail)
    return response
