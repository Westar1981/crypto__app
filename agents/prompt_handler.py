from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class FileChange:
    filepath: str
    description: str
    changes: List[str]
    original: Optional[str] = None

class PromptHandler:
    def __init__(self):
        self.file_changes: Dict[str, FileChange] = {}
        
    def analyze_prompt(self, prompt: str) -> Dict[str, any]:
        """Analyze prompt content and required changes"""
        # ...existing code...
        
    def format_response(self, changes: List[FileChange]) -> str:
        """Format response according to prompt requirements"""
        response = []
        for change in changes:
            response.append(f"### {change.filepath}\n")
            response.append(f"{change.description}\n")
            response.append("<file>")
            if change.filepath.endswith('.md'):
                response.append("````markdown")
            else:
                response.append("```python")
            response.append(f"// filepath: {change.filepath}")
            response.append(change.changes)
            response.append("```" + "`" * (4 if change.filepath.endswith('.md') else 0))
            response.append("</file>\n")
        return "\n".join(response)
