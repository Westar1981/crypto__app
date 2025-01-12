from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Deque
from collections import deque
import asyncio
import logging
import time
import heapq

from .learner_agent import MetaLearner, CapabilityEvolution, LearnerConfig
from .prompt_handler import PromptHandler
from ..core.types import FileChange
from ..core.memory_handler import MemoryHandler, MemoryConfig
from ..core.metrics_tracker import MetricsTracker, MetricsConfig
from ..core.self_monitor import SelfMonitor, MonitorConfig

logger = logging.getLogger(__name__)

@dataclass
class PromptPriority:
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3

@dataclass
class ManagerConfig:
    """Configuration for interaction manager"""
    min_state_size: int = 32
    max_state_size: int = 512
    default_state_size: int = 64
    min_action_size: int = 16
    max_action_size: int = 256
    default_action_size: int = 32
    batch_timeout: float = 0.1
    max_batch_size: int = 32
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000
    max_queue_size: int = 1000
    priority_levels: int = 4
    dynamic_batch_timeout: bool = True
    min_batch_timeout: float = 0.01
    max_batch_timeout: float = 1.0

class AgentInteractionManager:
    def __init__(self, config: Optional[ManagerConfig] = None):
        self.config = config or ManagerConfig()
        self.prompt_handler = PromptHandler()
        self.meta_learner = None
        self.capability_evolution = CapabilityEvolution()
        self.prompt_queue: Deque[Tuple[str, asyncio.Future]] = deque()
        self.memory_handler = MemoryHandler(
            MemoryConfig(
                max_cache_entries=self.config.max_cache_size,
                cache_cleanup_interval=self.config.cache_ttl
            )
        )
        self.prompt_cache = self.memory_handler.get_cache('prompts')
        self.processing_task: Optional[asyncio.Task] = None
        self.metrics = MetricsTracker(
            MetricsConfig(window_size=1000, log_interval=60)
        )
        self.priority_queues = [
            [] for _ in range(self.config.priority_levels)
        ]
        self.current_batch_size = self.config.max_batch_size
        self.monitor = SelfMonitor(
            MonitorConfig(
                check_interval=5.0,
                adaptation_rate=0.15,
                callbacks=[self._handle_adaptation]
            )
        )
        asyncio.create_task(self.monitor.start())
        
    def _validate_sizes(self, state_size: int, action_size: int) -> Tuple[int, int]:
        """Validate and adjust state/action sizes"""
        state_size = max(min(state_size, self.config.max_state_size), self.config.min_state_size)
        action_size = max(min(action_size, self.config.max_action_size), self.config.min_action_size)
        return state_size, action_size

    async def handle_prompt(self, prompt: str, priority: int = PromptPriority.NORMAL) -> str:
        start_time = time.time()
        try:
            cached = self.prompt_cache.get(hash(prompt))
            self.metrics.record_cache_access(cached is not None)
            
            future = asyncio.Future()
            self.priority_queues[priority].append((prompt, future))
            
            if not self.processing_task or self.processing_task.done():
                self.processing_task = asyncio.create_task(self._process_batch())
                
            result = await future
            self.metrics.record_latency(time.time() - start_time)
            return result
        except Exception as e:
            self.metrics.record_error()
            raise
        
    async def _process_batch(self):
        """Process prompts with priority handling"""
        while any(self.priority_queues):
            batch = []
            futures = []
            
            # Collect items from priority queues
            for queue in self.priority_queues:
                while queue and len(batch) < self.current_batch_size:
                    prompt, future = queue.pop(0)
                    batch.append(prompt)
                    futures.append(future)
                if len(batch) >= self.current_batch_size:
                    break

            # Dynamic timeout adjustment
            if self.config.dynamic_batch_timeout:
                timeout = self._calculate_batch_timeout(len(batch))
                await asyncio.sleep(timeout)

            if not batch:
                break
                
            try:
                states = await self.meta_learner.batch_process_states(
                    [self.prompt_handler.analyze_prompt(p) for p in batch]
                )
                actions = await self.meta_learner.get_action(states)
                
                for i, (prompt, future) in enumerate(zip(batch, futures)):
                    try:
                        changes = self.capability_evolution.action_to_changes(actions[i])
                        result = self.prompt_handler.format_response(changes)
                        
                        cache_key = hash(prompt)
                        self.prompt_cache.put(cache_key, result)
                        
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                        
                self.metrics.record_latency(
                    time.time() - min(f.start_time for f in futures)
                )
                        
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                        
            # Adjust batch size based on metrics
            self._adjust_batch_size()

    def _calculate_batch_timeout(self, current_batch: int) -> float:
        """Calculate dynamic batch timeout"""
        if current_batch >= self.current_batch_size:
            return self.config.min_batch_timeout
            
        fill_rate = len(self.prompt_queue) / self.config.max_queue_size
        return max(
            self.config.min_batch_timeout,
            min(
                self.config.max_batch_timeout,
                (1 - fill_rate) * self.config.max_batch_timeout
            )
        )

    def _adjust_batch_size(self):
        """Adjust batch size based on performance metrics"""
        stats = self.metrics.get_stats()
        if 'latency' in stats:
            p95_latency = stats['latency']['p95']
            
            # Increase batch size if performance is good
            if p95_latency < 0.5:  # Under 500ms
                self.current_batch_size = min(
                    self.current_batch_size * 2,
                    self.config.max_batch_size
                )
            # Decrease if too slow
            elif p95_latency > 1.0:  # Over 1 second
                self.current_batch_size = max(
                    self.current_batch_size // 2,
                    1
                )
                        
    def _cleanup_cache(self):
        """No longer needed - handled by MemoryHandler"""
        pass

    def _handle_adaptation(self, action: str):
        """Handle adaptation requests"""
        adaptations = {
            'reduce_cpu_load': self._reduce_batch_size,
            'reduce_memory': self._cleanup_caches,
            'optimize_latency': self._optimize_timeouts,
            'improve_throughput': self._increase_batch_size
        }
        
        if action in adaptations:
            adaptations[action]()
            
    def _optimize_timeouts(self):
        """Optimize batch timeouts"""
        self.config.batch_timeout = max(
            self.config.batch_timeout / 2,
            self.config.min_batch_timeout
        )
