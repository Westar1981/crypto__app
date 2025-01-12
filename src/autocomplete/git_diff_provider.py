from .types import CompletionContext, CompletionMatch, CompletionPlan, BaseCompletionProvider

class GitDiffCompletionProvider(BaseCompletionProvider):
    """Provides completions for git diff commands"""

    def __init__(self):
        super().__init__(CompletionPlan(
            prefix="gd",
            commands={
                "gd": "git diff",
                "gds": "git diff --staged",
                "gdh": "git diff HEAD",
                "gdc": "git diff --cached",
                "gdw": "git diff --word-diff",
                "gdn": "git diff --name-only",
            },
            description="Git diff commands",
            priority=100
        ))

    def get_completions(self, ctx: CompletionContext) -> List[CompletionMatch]:
        if not ctx.prefix or not ctx.prefix.startswith(self.plan.prefix):
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