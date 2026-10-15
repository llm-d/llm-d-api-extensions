# SPDX-License-Identifier: Apache-2.0
import pytest

from llm_d_api_extensions_vllm.server_introspection.common import reset_drift_events


@pytest.fixture(autouse=True)
def _clean_drift_events():
    # Drift events are process wide, so one test's would leak into the next.
    reset_drift_events()
    yield
    reset_drift_events()
