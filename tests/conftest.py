# SPDX-FileCopyrightText: 2026 Paul Burg
# SPDX-License-Identifier: Apache-2.0
import pytest

from tll.sim import World


@pytest.fixture(scope="session")
def world():
    return World()
