from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import asyncio
import time
import psutil
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class MonitorConfig:
    check_interval: float = 5.0 
    history_size: int = 100
    alert_threshold: float = 0.85
    adaptation_rate: float = 0.1
    performance_window: int = 20
    callbacks: List[Callable] = field(default_factory=list)

class SelfMonitor:
    """Monitors system performance and adapts behavior"""
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        self.metrics = {
            'cpu_usage': deque(maxlen=self.config.history_size),
            'memory_usage': deque(maxlen=self.config.history_size),
            'latency': deque(maxlen=self.config.history_size),
            'throughput': deque(maxlen=self.config.history_size)
        }
        self.running = False
        self._monitor_task = None
        
    async def start(self):
        """Start monitoring"""
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self):
        """Stop monitoring"""
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._check_metrics()
                await self._analyze_performance()
                await asyncio.sleep(self.config.check_interval)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                
    async def _check_metrics(self):
        """Collect current metrics"""
        try:
            # System metrics
            cpu = psutil.cpu_percent() / 100.0
            memory = psutil.Process().memory_info().rss / psutil.virtual_memory().total
            
            self.metrics['cpu_usage'].append(cpu)
            self.metrics['memory_usage'].append(memory)
            
            self._notify_if_threshold_exceeded()
            
        except Exception as e:
            logger.error(f"Metric collection error: {e}")
            
    async def _analyze_performance(self):
        """Analyze performance trends"""
        window = self.config.performance_window
        for metric, values in self.metrics.items():
            if len(values) < window:
                continue
                
            recent = list(values)[-window:]
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            
            if abs(trend) > self.config.adaptation_rate:
                await self._adapt_behavior(metric, trend)
                
    async def _adapt_behavior(self, metric: str, trend: float):
        """Adapt system behavior based on trends"""
        adaptations = {
            'cpu_usage': self._adapt_cpu_usage,
            'memory_usage': self._adapt_memory_usage,
            'latency': self._adapt_latency,
            'throughput': self._adapt_throughput
        }
        
        if metric in adaptations:
            await adaptations[metric](trend)
            
    async def _adapt_cpu_usage(self, trend: float):
        """Adapt to CPU usage trends"""
        if trend > 0:  # Usage increasing
            logger.info("Reducing batch sizes due to CPU pressure")
            # Signal components to reduce load
            await self._notify_adaptation('reduce_cpu_load')
            
    async def _adapt_memory_usage(self, trend: float):
        """Adapt to memory usage trends"""
        if trend > 0:  # Usage increasing
            logger.info("Triggering memory cleanup")
            # Signal components to free memory
            await self._notify_adaptation('reduce_memory')
            
    async def _adapt_latency(self, trend: float):
        """Adapt to latency trends"""
        if trend > 0:  # Latency increasing
            logger.info("Adjusting processing to reduce latency")
            await self._notify_adaptation('optimize_latency')
            
    async def _adapt_throughput(self, trend: float):
        """Adapt to throughput trends"""
        if trend < 0:  # Throughput decreasing
            logger.info("Adjusting to improve throughput")
            await self._notify_adaptation('improve_throughput')
            
    def _notify_if_threshold_exceeded(self):
        """Check for threshold violations"""
        for metric, values in self.metrics.items():
            if not values:
                continue
                
            current = values[-1]
            if current > self.config.alert_threshold:
                logger.warning(f"{metric} exceeded threshold: {current:.2f}")
                for callback in self.config.callbacks:
                    try:
                        callback(metric, current)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                        
    async def _notify_adaptation(self, action: str):
        """Notify system of required adaptations"""
        for callback in self.config.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(action)
                else:
                    callback(action)
            except Exception as e:
                logger.error(f"Adaptation callback error: {e}")
