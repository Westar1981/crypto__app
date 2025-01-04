import unittest
from core.analysis.blackbox_analyzer import BlackboxAnalyzer
from core.analysis.cot_analyzer import CoTAnalyzer

class TestBlackboxAnalyzer(unittest.TestCase):
    def setUp(self):
        self.cot_analyzer = CoTAnalyzer()
        self.blackbox_analyzer = BlackboxAnalyzer(self.cot_analyzer)

    def test_build_context(self):
        response = "The AI suggests optimizing the algorithm for better performance."
        context = self.blackbox_analyzer.build_context(response)
        self.assertIn(response, self.blackbox_analyzer.context_cache)
        self.assertEqual(context, {"key_info": response})

    def test_generate_prompt(self):
        context = {"key_info": "Optimize algorithm."}
        prompt = self.blackbox_analyzer.generate_prompt(context)
        self.assertEqual(prompt, "Based on the context: Optimize algorithm., what are the next steps?")

    def test_analyze_response(self):
        response = "The AI suggests optimizing the algorithm for better performance."
        prompt = self.blackbox_analyzer.analyze_response(response)
        self.assertIn("Based on the context:", prompt)

if __name__ == "__main__":
    unittest.main()
