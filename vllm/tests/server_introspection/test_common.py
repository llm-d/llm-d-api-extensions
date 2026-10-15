# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the vLLM drift tolerance helpers in `common.py`."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
from starlette.datastructures import State

from llm_d_api_extensions_vllm.server_introspection.common import (
    MISSING,
    build_tolerant,
    drift_events,
    guarded_init_state,
    read_attr,
    report_drift,
)


class _Model(BaseModel):
    count: int | None = None
    label: str | None = None
    required: int = 0


class _Strict(BaseModel):
    count: int


class TestReadAttr:
    def test_returns_present_value(self):
        assert read_attr(SimpleNamespace(a=1), "a") == 1
        assert drift_events() == []

    def test_present_none_is_not_drift(self):
        assert read_attr(SimpleNamespace(a=None), "a", MISSING) is None
        assert drift_events() == []

    def test_missing_returns_default_and_reports(self):
        assert read_attr(SimpleNamespace(), "gone") is None
        assert read_attr(SimpleNamespace(), "gone", MISSING) is MISSING
        assert drift_events() == ["SimpleNamespace.gone is not available"]

    def test_none_owner_returns_default_silently(self):
        assert read_attr(None, "anything", 7) == 7
        assert drift_events() == []

    def test_property_that_raises_is_drift(self):
        class Cfg:
            @property
            def broken(self):
                raise RuntimeError("boom")

        assert read_attr(Cfg(), "broken") is None
        assert drift_events() == ["Cfg.broken is not available"]

    def test_module_owner_is_named_by_module(self):
        import vllm.envs as envs

        assert read_attr(envs, "VLLM_NO_SUCH_SETTING") is None
        assert drift_events() == ["vllm.envs.VLLM_NO_SUCH_SETTING is not available"]


class TestReportDrift:
    def test_each_event_recorded_once(self):
        report_drift("a")
        report_drift("b")
        report_drift("a")
        assert drift_events() == ["a", "b"]


class TestBuildTolerant:
    def test_valid_values_pass_through(self):
        assert build_tolerant(_Model, count=1, label="x") == _Model(count=1, label="x")
        assert drift_events() == []

    def test_invalid_field_is_dropped_and_others_kept(self):
        result = build_tolerant(_Model, count="not a number", label="x", required=3)
        assert result == _Model(count=None, label="x", required=3)
        assert drift_events() == ["_Model.count has unexpected value 'not a number'"]

    def test_invalid_field_with_default_takes_the_default(self):
        assert build_tolerant(_Model, required="nope").required == 0

    def test_invalid_field_without_default_still_raises(self):
        with pytest.raises(ValidationError):
            build_tolerant(_Strict, count="nope")


class _Plugin:
    def __init__(self, error: Exception | None = None):
        self.error = error

    @guarded_init_state("thing_response", "thing")
    async def init_state(self, engine_client, state: State, args: Namespace) -> None:
        if self.error is not None:
            raise self.error
        state.thing_response = "ok"


class TestGuardedInitState:
    def test_success_is_untouched(self):
        state = FastAPI().state
        asyncio.run(_Plugin().init_state(None, state, Namespace()))
        assert state.thing_response == "ok"
        assert not hasattr(state, "thing_unavailable_detail")

    def test_exception_is_contained_and_marks_endpoint_unavailable(self):
        state = FastAPI().state
        asyncio.run(_Plugin(AttributeError("no such field")).init_state(None, state, Namespace()))
        assert state.thing_response is None
        assert "AttributeError" in state.thing_unavailable_detail

    def test_cancellation_is_not_swallowed(self):
        # Server shutdown during startup must still propagate.
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(_Plugin(asyncio.CancelledError()).init_state(None, State(), Namespace()))
