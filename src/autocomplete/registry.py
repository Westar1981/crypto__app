from typing import List
from .types import CompletionProvider
from .git_diff_provider import GitDiffCompletionProvider

class CompletionRegistry:
    def __init__(self):
        self.providers: List[CompletionProvider] = []

    def register_provider(self, provider: CompletionProvider):
        self.providers.append(provider)
        self.providers.sort(key=lambda p: p.get_priority(), reverse=True)

    def get_completions(self, ctx):
        for provider in self.providers:
            if provider.supports_prefix(ctx.prefix):
                return provider.get_completions(ctx)
        return []

# Usage
registry = CompletionRegistry()
registry.register_provider(GitDiffCompletionProvider())