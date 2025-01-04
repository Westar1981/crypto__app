from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
import re
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Rule:
    name: str
    conditions: List[str]
    consequences: List[str]
    priority: int = 0
    metadata: Dict[str, Any] = None
    dependencies: Set[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.dependencies is None:
            self.dependencies = set()
            self._extract_dependencies()
    
    def _extract_dependencies(self):
        """Extract rule dependencies from conditions."""
        for condition in self.conditions:
            if isinstance(condition, str):
                terms = condition.split('(')[0].strip()
                self.dependencies.add(terms)

class PrologEngine:
    def __init__(self):
        self.rules: List[Rule] = []
        self.facts: Set[str] = set()
        self.patterns: Dict[str, re.Pattern] = {}
        self.rule_index: Dict[str, List[Rule]] = {}
        self.query_cache: Dict[str, bool] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def add_rule(self, rule: Rule) -> None:
        """Add a new rule with indexing."""
        self.rules.append(rule)
        # Sort rules by priority
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        # Index rule by consequences
        for consequence in rule.consequences:
            pred = self._get_predicate(consequence)
            if pred not in self.rule_index:
                self.rule_index[pred] = []
            self.rule_index[pred].append(rule)
            
    def add_fact(self, fact: str) -> None:
        """Add a new fact to the knowledge base."""
        self.facts.add(fact)
        
    def add_pattern(self, name: str, pattern: str) -> None:
        """Add a new pattern for matching."""
        try:
            self.patterns[name] = re.compile(pattern)
        except re.error as e:
            logger.error(f"Invalid pattern {pattern}: {str(e)}")
            raise
            
    def _get_predicate(self, term: str) -> str:
        """Extract predicate from term."""
        return term.split('(')[0].strip()
        
    def query(self, goal: str) -> bool:
        """Enhanced query with caching and optimization."""
        # Check cache
        if goal in self.query_cache:
            return self.query_cache[goal]
            
        result = self._optimized_query(goal)
        self.query_cache[goal] = result
        return result
        
    def _optimized_query(self, goal: str) -> bool:
        """Optimized query execution."""
        if goal in self.facts:
            return True
            
        pred = self._get_predicate(goal)
        if pred in self.rule_index:
            relevant_rules = self.rule_index[pred]
        else:
            relevant_rules = [r for r in self.rules if self._matches_pattern(goal, r.consequences)]
            
        return any(
            all(self.query(condition) for condition in rule.conditions)
            for rule in relevant_rules
        )
        
    async def parallel_query(self, goals: List[str]) -> Dict[str, bool]:
        """Execute multiple queries in parallel."""
        results = {}
        async with ThreadPoolExecutor() as executor:
            futures = {goal: executor.submit(self.query, goal) for goal in goals}
            for goal, future in futures.items():
                try:
                    results[goal] = future.result()
                except Exception as e:
                    logger.error(f"Error processing query {goal}: {str(e)}")
                    results[goal] = False
        return results
        
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build rule dependency graph."""
        graph = {}
        for rule in self.rules:
            for dep in rule.dependencies:
                if dep not in graph:
                    graph[dep] = set()
                for cons in rule.consequences:
                    pred = self._get_predicate(cons)
                    graph[dep].add(pred)
        return graph
        
    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[Rule]:
        """Sort rules based on dependencies."""
        visited = set()
        sorted_rules = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph.get(node, []):
                visit(neighbor)
            relevant_rules = [r for r in self.rules 
                            if any(node == self._get_predicate(c) 
                                for c in r.consequences)]
            sorted_rules.extend(relevant_rules)
            
        for node in graph:
            visit(node)
            
        return sorted_rules
        
    def apply_rules(self) -> Set[str]:
        """Apply all rules to derive new facts."""
        new_facts = set()
        changed = True
        
        while changed:
            changed = False
            for rule in self.rules:
                if all(self.query(condition) for condition in rule.conditions):
                    for consequence in rule.consequences:
                        if consequence not in self.facts:
                            new_facts.add(consequence)
                            changed = True
                            
        return new_facts

cot_analyzer = ChainOfThoughtAnalyzer()
root = cot_analyzer.create_thought_chain("analysis_root", "Code Analysis")
thought = cot_analyzer.add_thought("analysis_root", "pattern_1", "Singleton Pattern", 0.95)
