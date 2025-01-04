from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import asyncio
from loguru import logger

@dataclass
class PipelineStage:
    name: str
    processor: Callable
    requirements: Dict[str, Any]
    cache_enabled: bool = True

class PipelineManager:
    def __init__(self):
        self.pipelines: Dict[str, List[PipelineStage]] = {}
        self.result_cache: Dict[str, Any] = {}
        
    def create_pipeline(self, name: str) -> None:
        """Create a new analysis pipeline."""
        if name in self.pipelines:
            raise ValueError(f"Pipeline {name} already exists")
        self.pipelines[name] = []
        
    def add_stage(self, pipeline_name: str, stage: PipelineStage) -> None:
        """Add a processing stage to a pipeline."""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} does not exist")
        self.pipelines[pipeline_name].append(stage)
        
    async def execute_pipeline(self, pipeline_name: str, input_data: Any) -> Any:
        """Execute all stages in a pipeline."""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} does not exist")
            
        result = input_data
        cache_key = f"{pipeline_name}:{hash(str(input_data))}"
        
        # Check cache first
        if cache_key in self.result_cache:
            logger.info(f"Cache hit for {pipeline_name}")
            return self.result_cache[cache_key]
            
        try:
            for stage in self.pipelines[pipeline_name]:
                result = await self._execute_stage(stage, result)
                
            # Cache result if all stages are cache-enabled
            if all(stage.cache_enabled for stage in self.pipelines[pipeline_name]):
                self.result_cache[cache_key] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error executing pipeline {pipeline_name}: {str(e)}")
            raise
            
    async def _execute_stage(self, stage: PipelineStage, input_data: Any) -> Any:
        """Execute a single pipeline stage."""
        try:
            logger.info(f"Executing stage: {stage.name}")
            result = await stage.processor(input_data, **stage.requirements)
            return result
        except Exception as e:
            logger.error(f"Error in stage {stage.name}: {str(e)}")
            raise
            
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self.result_cache.clear()
