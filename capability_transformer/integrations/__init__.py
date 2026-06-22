"""Agent-framework integrations for the capability gateway."""

__all__ = ["CapabilityGuard"]


def __getattr__(name):
    # Lazy import so the optional langchain dependency is only required on use.
    if name == "CapabilityGuard":
        from .langchain_guard import CapabilityGuard

        return CapabilityGuard
    raise AttributeError(name)
