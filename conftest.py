from __future__ import annotations

import shutil
import sys
import types

try:
    import agent.memory_provider  # noqa: F401
except ImportError:
    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    memory_provider.MemoryProvider = MemoryProvider
    agent.memory_provider = memory_provider
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = memory_provider


def pytest_sessionfinish(session, exitstatus):
    for name in (".pytest_cache",):
        shutil.rmtree(name, ignore_errors=True)
