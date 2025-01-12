from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FileChange:
    """Represents a change to be made to a file"""
    filepath: str
    content: str
    metadata: Dict[str, Any] = None
