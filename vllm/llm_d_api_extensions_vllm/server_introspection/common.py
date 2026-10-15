# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the server introspection endpoint plugins.

Everything these plugins read from vLLM is internal API that can change between
releases. vLLM runs `init_state` with no exception handling which means a plugin
that raises there aborts server startup. The helpers here keep that from
happening in three layers:

- `read_attr` reads one vLLM attribute and yields a default when it is gone.
- `build_tolerant` nulls individual fields whose value no longer fits the
  response schema instead of failing the whole response.
- `guarded_init_state` is the backstop. Anything still raised from
  `init_state` is logged and turns the endpoint into a `503`.

Every degradation is recorded through `report_drift` (logged once, listed by
`drift_events()`) which is what the contract tests assert on.
"""

import functools
import logging
from argparse import Namespace
from collections.abc import Awaitable, Callable
from types import ModuleType
from typing import Any, TypeVar

from fastapi import HTTPException, Request
from pydantic import BaseModel, ValidationError
from starlette.datastructures import State

try:
    from vllm.logger import init_logger
except Exception:  # vLLM moved or removed it
    init_logger = None

try:
    from vllm.tasks import GENERATION_TASKS, POOLING_TASKS
except Exception:  # vLLM moved or removed them
    GENERATION_TASKS = POOLING_TASKS = None

# Snapshot of vLLM's task names, used only when `vllm.tasks` no longer exports
# them. A plugin loads when `required_tasks` intersects the server's tasks, so
# a stale name here costs nothing.
_FALLBACK_ENGINE_TASKS: tuple[str, ...] = (
    "generate",
    "transcription",
    "realtime",
    "embed",
    "classify",
    "token_embed",
    "token_classify",
    "plugin",
    "embed&token_classify",
)

# Tasks for plugins that need an engine. Excludes the `render` frontend task
# because there is nothing to introspect on the CPU only render server.
ENGINE_TASKS: tuple[str, ...] = (
    tuple(GENERATION_TASKS) + tuple(POOLING_TASKS)
    if GENERATION_TASKS is not None and POOLING_TASKS is not None
    else _FALLBACK_ENGINE_TASKS
)

_UNSET = object()

# Default for `read_attr` callers that must tell "attribute is gone" apart
# from "attribute is None".
MISSING: Any = object()

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def get_logger(name: str) -> logging.Logger:
    # "vllm." prefix required. vLLM's default logging config only attaches a
    # handler to the "vllm" logger tree (propagate=False), so a bare __name__
    # logger has no handler anywhere and silently drops every message.
    if init_logger is None:
        return logging.getLogger(f"vllm.{name}")
    return init_logger(f"vllm.{name}")


logger = get_logger(__name__)

# Insertion ordered so `drift_events()` reads in the order things were found.
_drift_events: dict[str, None] = {}


def report_drift(event: str) -> None:
    """Record one way the running vLLM differs from what the plugins expect.

    Logged as a warning the first time each distinct `event` is seen.
    """
    if event in _drift_events:
        return
    _drift_events[event] = None
    logger.warning(
        "server introspection: %s. This vLLM version differs from the one this "
        "package was written against, the affected field is reported as null.",
        event,
    )


def drift_events() -> list[str]:
    return list(_drift_events)


def reset_drift_events() -> None:
    _drift_events.clear()


if init_logger is None:
    report_drift("vllm.logger.init_logger is not importable")
if GENERATION_TASKS is None or POOLING_TASKS is None:
    report_drift("vllm.tasks.GENERATION_TASKS/POOLING_TASKS are not importable")


def read_attr(obj: Any, name: str, default: Any = None) -> Any:
    """`getattr` for vLLM objects that reports a missing attribute as drift.

    `obj` being `None` returns `default` silently. Its own absence was
    already reported when the caller read it.
    """
    if obj is None:
        return default
    try:
        return getattr(obj, name)
    except Exception:
        owner = obj.__name__ if isinstance(obj, ModuleType) else type(obj).__name__
        report_drift(f"{owner}.{name} is not available")
        return default


def build_tolerant(model_cls: type[_ModelT], **values: Any) -> _ModelT:
    """Build `model_cls`, dropping any field whose value fails validation.

    A dropped field takes its schema default (`None` for nearly all of them).
    Raises `ValidationError` only when a field that failed has no default.
    """
    try:
        return model_cls(**values)
    except ValidationError as e:
        invalid = {err["loc"][0] for err in e.errors() if err["loc"]}
        for field in sorted(map(str, invalid)):
            report_drift(f"{model_cls.__name__}.{field} has unexpected value {values.get(field)!r}")
        return model_cls(**{k: v for k, v in values.items() if k not in invalid})


def no_engine_detail(endpoint: str) -> str:
    return f"{endpoint} requires an engine, which this server does not have"


def _unavailable_detail_attr(state_attr: str) -> str:
    return f"{state_attr.removesuffix('_response')}_unavailable_detail"


_InitState = Callable[[Any, Any, State, Namespace], Awaitable[None]]


def guarded_init_state(state_attr: str, endpoint: str) -> Callable[[_InitState], _InitState]:
    """Keep a failing `init_state` from aborting vLLM server startup.

    On any exception the failure is logged, `state.<state_attr>` is set to
    `None` and the endpoint serves `503` with the reason (see
    `cached_response`).
    """

    def decorator(init_state: _InitState) -> _InitState:
        @functools.wraps(init_state)
        async def wrapper(self: Any, engine_client: Any, state: State, args: Namespace) -> None:
            try:
                await init_state(self, engine_client, state, args)
            except Exception as e:
                logger.exception(
                    "%s plugin failed to initialize, its endpoint will return 503. "
                    "This usually means the vLLM version changed an interface the "
                    "plugin reads.",
                    endpoint,
                )
                setattr(state, state_attr, None)
                setattr(
                    state,
                    _unavailable_detail_attr(state_attr),
                    f"{endpoint} is unavailable: plugin initialization failed against "
                    f"this vLLM version ({type(e).__name__}), see the server log",
                )

        return wrapper

    return decorator


def cached_response(request: Request, state_attr: str, unavailable_detail: str):
    """Return the response `init_state` cached on `app.state.<state_attr>`.

    Raises 500 when `init_state` never ran and 503 when it ran but cached
    `None` (the endpoint can't serve on this server). The 503 detail is the
    reason `init_state` left on `app.state`, else `unavailable_detail`.
    """
    response = getattr(request.app.state, state_attr, _UNSET)
    if response is _UNSET:
        raise HTTPException(
            status_code=500,
            detail=f"{state_attr.removesuffix('_response')} plugin state was never initialized",
        )
    if response is None:
        detail = getattr(request.app.state, _unavailable_detail_attr(state_attr), None)
        raise HTTPException(status_code=503, detail=detail or unavailable_detail)
    return response
