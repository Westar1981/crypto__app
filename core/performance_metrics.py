
class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}

    def track_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def report_metrics(self):
        for name, values in self.metrics.items():
            average = sum(values) / len(values)
            print(f"{name}: {average}")

# Example usage
metrics = PerformanceMetrics()
metrics.track_metric('response_time', 200)
metrics.track_metric('response_time', 220)
metrics.track_metric('accuracy', 95)
metrics.track_metric('accuracy', 97)
metrics.report_metrics()