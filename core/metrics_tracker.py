from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import time
import psutil
import numpy as np
from collections import deque
import threading
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class MetricsConfig:
    window_size: int = 100  # Size of sliding window
    log_interval: int = 60  # Seconds between logging
    alert_threshold: float = 0.95  # Performance alert threshold
    percentiles: List[float] = field(default_factory=lambda: [50, 90, 95, 99])
    trend_window: int = 10  # Window for trend analysis
    prediction_horizon: int = 5  # Steps to predict ahead
    adaptive_threshold: bool = True
    trend_sensitivity: float = 0.1
    alert_callbacks: List[Callable] = field(default_factory=list)
    resource_warning_threshold: float = 0.8

class MetricsTracker:
    """Tracks system performance metrics"""
    
    def __init__(self, config: Optional<MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.metrics: Dict[str, deque] = {
            'latency': deque(maxlen=self.config.window_size),
            'throughput': deque(maxlen=self.config.window_size),
            'memory_usage': deque(maxlen=self.config.window_size),
            'cpu_usage': deque(maxlen=self.config.window_size),
            'cache_hits': deque(maxlen=self.config.window_size),
            'cache_misses': deque(maxlen=self.config.window_size),
            'error_rate': deque(maxlen=self.config.window_size)
        }
        self.trends = {
            metric: deque(maxlen=self.config.trend_window) 
            for metric in self.metrics
        }
        self.adaptive_thresholds = {
            metric: self._init_threshold(metric)
            for metric in self.metrics
        }
        self.start_time = time.time()
        self._lock = threading.Lock()
        
    def _init_threshold(self, metric: str) -> float:
        """Initialize adaptive threshold for metric"""
        base_thresholds = {
            'latency': 1.0,
            'error_rate': 0.05,
            'memory_usage': 0.8,
            'cpu_usage': 0.9,
            'cache_hits': 0.5
        }
        return base_thresholds.get(metric, 0.95)
        
    def record_latency(self, latency: float) -> None:
        """Record request latency"""
        self.record_metric('latency', latency)
            
    def record_cache_access(self, hit: bool) -> None:
        """Record cache hit/miss"""
        metric = 'cache_hits' if hit else 'cache_misses'
        self.record_metric(metric, 1)
            
    def record_error(self) -> None:
        """Record error occurrence"""
        with self._lock:
            total_requests = len(self.metrics['latency'])
            if total_requests > 0:
                self.metrics['error_rate'].append(1/total_requests)
                
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics"""
        with self._lock:
            stats = {}
            for metric, values in self.metrics.items():
                if not values:
                    continue
                    
                percentiles = np.percentile(list(values), self.config.percentiles)
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    **{f'p{p}': v for p, v in zip(self.config.percentiles, percentiles)}
                }
            return stats
            
    def analyze_trend(self, metric: str) -> Tuple[float, float]:
        """Analyze metric trend using linear regression"""
        if len(self.trends[metric]) < 3:
            return 0.0, 0.0
            
        x = np.arange(len(self.trends[metric]))
        y = np.array(self.trends[metric])
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        
        return slope, r_value

    def predict_value(self, metric: str, steps: int = None) -> float:
        """Predict future metric value"""
        steps = steps or self.config.prediction_horizon
        slope, r_value = self.analyze_trend(metric)
        
        if abs(r_value) < 0.5:  # Poor correlation
            return float('nan')
            
        current = self.metrics[metric][-1] if self.metrics[metric] else 0
        return current + (slope * steps)

    def check_health(self) -> Tuple[bool, List[str]]:
        """Enhanced health check with predictions"""
        warnings = []
        stats = self.get_stats()

        for metric, values in stats.items():
            # Current value check
            current = values['mean']
            threshold = self.adaptive_thresholds[metric]
            
            if current > threshold:
                warnings.append(f'High {metric}: {current:.2f} > {threshold:.2f}')
                
            # Trend analysis
            slope, r_value = self.analyze_trend(metric)
            if abs(r_value) > 0.7:  # Strong correlation
                if slope > self.config.trend_sensitivity:
                    predicted = self.predict_value(metric)
                    if not np.isnan(predicted) and predicted > threshold:
                        warnings.append(
                            f'Warning: {metric} trending up, predicted: {predicted:.2f}'
                        )

        # Resource usage projection
        memory_trend = self.analyze_trend('memory_usage')[0]
        if memory_trend > 0:
            projected_memory = self.predict_value('memory_usage')
            if projected_memory > self.config.resource_warning_threshold:
                warnings.append(
                    f'Memory usage projected to reach {projected_memory:.1%}'
                )

        # Notify callbacks
        if warnings and self.config.alert_callbacks:
            for callback in self.config.alert_callbacks:
                try:
                    callback(warnings)
                except Exception as e:
                    logging.error(f"Alert callback failed: {e}")

        return len(warnings) == 0, warnings

    def update_thresholds(self):
        """Adaptively update metric thresholds"""
        if not self.config.adaptive_threshold:
            return
            
        for metric, values in self.metrics.items():
            if len(values) < self.config.window_size // 2:
                continue
                
            # Calculate new threshold using standard deviation
            mean = np.mean(values)
            std = np.std(values)
            new_threshold = mean + (2 * std)  # 95% confidence interval
            
            # Smooth threshold update
            current = self.adaptive_thresholds[metric]
            self.adaptive_thresholds[metric] = (0.8 * current) + (0.2 * new_threshold)

    def record_metric(self, metric: str, value: float):
        """Generic metric recording with trend tracking"""
        with self._lock:
            self.metrics[metric].append(value)
            self.trends[metric].append(value)
            
        if len(self.trends[metric]) >= self.config.trend_window:
            self.update_thresholds()

    def log_metrics(self) -> None:
        """Log current metrics"""
        stats = self.get_stats()
        for metric, values in stats.items():
            logger.info(f"{metric}: mean={values['mean']:.3f}, p95={values['p95']:.3f}")
