from typing import Dict, List, Tuple, Any
import re

class PromptAnalyzer:
    def __init__(self):
        self.file_pattern = re.compile(r'<file>.*?</file>', re.DOTALL)
        
    def extract_file_blocks(self, content: str) -> List[Tuple[str, str]]:
        """Extract file blocks from content"""
        matches = self.file_pattern.findall(content)
        blocks = []
        for match in matches:
            filepath = self._extract_filepath(match)
            if filepath:
                blocks.append((filepath, match))
        return blocks
        
    def _extract_filepath(self, block: str) -> str:
        """Extract filepath from code block."""
        lines = block.split('\n')
        for line in lines:
            if '// filepath:' in line or '# filepath:' in line:
                return line.split('filepath:')[1].strip()
        return ""
        
    def parse_requirements(self, prompt: str) -> Dict[str, Any]:
        """Parse requirements from prompt."""
        requirements = {
            'files': [],
            'changes': [],
            'constraints': []
        }
        
        # Extract file paths and changes
        for line in prompt.split('\n'):
            if line.startswith('###'):
                requirements['files'].append(line.strip('# '))
            elif line.startswith('- '):
                requirements['changes'].append(line.strip('- '))
                
        return requirements
