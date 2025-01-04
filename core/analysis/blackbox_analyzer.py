# Blackbox AI Analyzer for Contextual Chain of Thought Prompts

from core.analysis.cot_analyzer import CoTAnalyzer

class BlackboxAnalyzer:
    def __init__(self, cot_analyzer: CoTAnalyzer):
        self.cot_analyzer = cot_analyzer
        self.context_cache = {}

    def build_context(self, response):
        """Build context from the AI response for further analysis."""
        context = self.extract_context(response)
        self.context_cache[response] = context
        return context

    def extract_context(self, response):
        """Extract relevant context from the AI response."""
        return {"key_info": response}  # Simplified example

    def generate_prompt(self, context):
        """Generate a contextual prompt based on the provided context."""
        prompt = f"Based on the context: {context['key_info']}, what are the next steps?"
        return prompt

    def analyze_response(self, response):
        """Analyze the AI response and generate a contextual prompt."""
        context = self.build_context(response)
        prompt = self.generate_prompt(context)
        return prompt

# Example usage
if __name__ == "__main__":
    cot_analyzer = CoTAnalyzer()
    analyzer = BlackboxAnalyzer(cot_analyzer)
    response = "The AI suggests optimizing the algorithm for better performance."
    prompt = analyzer.analyze_response(response)
    print(prompt)
