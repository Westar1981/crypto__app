
class EvaluationCriteria:
    def __init__(self):
        self.criteria = {
            'agent_satisfaction': [],
            'performance_improvements': []
        }

    def add_agent_satisfaction(self, score):
        self.criteria['agent_satisfaction'].append(score)

    def add_performance_improvement(self, metric):
        self.criteria['performance_improvements'].append(metric)

    def evaluate(self):
        avg_satisfaction = sum(self.criteria['agent_satisfaction']) / len(self.criteria['agent_satisfaction'])
        avg_performance = sum(self.criteria['performance_improvements']) / len(self.criteria['performance_improvements'])
        return {'average_satisfaction': avg_satisfaction, 'average_performance': avg_performance}

# Example usage
evaluation = EvaluationCriteria()
evaluation.add_agent_satisfaction(4.5)
evaluation.add_agent_satisfaction(4.7)
evaluation.add_performance_improvement(0.95)
evaluation.add_performance_improvement(0.97)
print(evaluation.evaluate())