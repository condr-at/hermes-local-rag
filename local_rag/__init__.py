"""Local RAG MemoryProvider. Import the Hermes provider only when requested.

The inference daemon and standalone storage/CLI do not require Hermes core.
"""


def __getattr__(name):
    if name in {'LocalRagProvider', 'register'}:
        from .provider import LocalRagProvider, register
        return {'LocalRagProvider': LocalRagProvider, 'register': register}[name]
    raise AttributeError(name)
