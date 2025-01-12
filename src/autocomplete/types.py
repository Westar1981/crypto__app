from typing import Dict, Protocol, Optional, TypeVar, Generic, List, Set
from dataclasses import dataclass, field

# ... (previous code remains unchanged)

class CompletionProvider(Protocol):
    """Interface for completion providers"""
    def get_completions(self, ctx: CompletionContext) -> List[CompletionMatch]: ...
    def supports_prefix(self, prefix: str) -> bool: ...
    def get_priority(self) -> int: ...
    def is_enabled(self) -> bool: ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...
    def get_plan(self) -> CompletionPlan: ...

class BaseCompletionProvider:
    """Base class for completion providers"""
    
    def __init__(self, plan: CompletionPlan):
        self.plan = plan

    def get_completions(self, ctx: CompletionContext) -> List[CompletionMatch]:
        if not self.supports_prefix(ctx.prefix):
            return []

        matches = []
        for shortcut, command in self.plan.commands.items():
            if shortcut.startswith(ctx.prefix) and shortcut not in ctx.seen:
                matches.append(CompletionMatch(
                    text=shortcut,
                    description=command,
                    score=len(ctx.prefix) / len(shortcut)
                ))
                ctx.seen.add(shortcut)

        return matches

    def supports_prefix(self, prefix: str) -> bool:
        return prefix.startswith(self.plan.prefix)

    def get_priority(self) -> int:
        return self.plan.priority

    def is_enabled(self) -> bool:
        return self.plan.enabled

    def enable(self) -> None:
        self.plan.enabled = True

    def disable(self) -> None:
        self.plan.enabled = False

    def get_plan(self) -> CompletionPlan:
        return self.plan