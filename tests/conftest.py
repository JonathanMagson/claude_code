"""Shared fixtures.

Generating a synthetic datacube is the slow part of any test run, so the two
sizes used across the suite are built once per session and shared.
"""

from __future__ import annotations

import pytest

from vegmon.config import demo_config
from vegmon.pipeline import run_pipeline
from vegmon.synthetic import generate_scene, scene_truth

# Small enough to keep the suite fast, large enough that several reference
# patches stay above the 0.5 ha minimum mapping unit after area scaling.
TEST_SIZE = 128


@pytest.fixture(scope="session")
def config():
    return demo_config()


@pytest.fixture(scope="session")
def scene(config):
    return generate_scene(config, size=TEST_SIZE, seed=7)


@pytest.fixture(scope="session")
def truth(scene):
    return scene_truth(scene)


@pytest.fixture(scope="session")
def tiny_scene(config):
    """A very small scene for tests that only need the plumbing to work."""
    return generate_scene(config, size=48, seed=11)


@pytest.fixture(scope="session")
def result(scene, config):
    return run_pipeline(scene, config)
