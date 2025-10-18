# src/agents/hypothesis_manager.py

import difflib
import time
import logging
import hashlib
import json
import re
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

@dataclass
class HypothesisMetrics:
    """Metrics for a hypothesis."""
    complexity: float = 0.0
    novelty: float = 0.0
    confidence: float = 0.0
    similarity_to_failed: float = 0.0
    approach_diversity: float = 0.0
    error_pattern_match: float = 0.0
    
    def get_overall_score(self) -> float:
        """Calculate overall score for ranking."""
        return (
            self.confidence * 0.3 +
            self.novelty * 0.25 +
            self.approach_diversity * 0.2 +
            (1 - self.similarity_to_failed) * 0.15 +
            (1 - self.complexity / 100) * 0.1  # Normalize complexity
        )


@dataclass
class HypothesisRecord:
    """Complete record of a hypothesis."""
    id: str
    hypothesis: str
    fixed_method: str
    changes: str
    approach_type: str
    timestamp: float
    execution_result: Optional[Dict] = None
    failure_reason: Optional[str] = None
    metrics: HypothesisMetrics = field(default_factory=HypothesisMetrics)
    parent_hypothesis_id: Optional[str] = None  # For tracking evolution


class HypothesisPool:
    """Enhanced hypothesis pool with diversity tracking and intelligent management."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.hypotheses: List[HypothesisRecord] = []
        self.logger = logging.getLogger(__name__)
        
        # Track patterns and approaches
        self.tried_approaches: Set[str] = set()
        self.error_patterns: Counter = Counter()
        self.successful_patterns: List[str] = []
        self.approach_taxonomy = ApproachTaxonomy()
        
        # Statistics
        self.total_attempts = 0
        self.success_count = 0
    
    def _remove_java_comments(self, code: str) -> str:
        """
        Remove single-line and multi-line comments from a Java method.
        
        Args:
            code (str): Java method code as a string.
        
        Returns:
            str: Code with comments removed.
        """
        # Remove multi-line comments (/* ... */)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Remove single-line comments (// ...)
        code = re.sub(r'//.*', '', code)
        
        # Strip extra whitespace from each line
        lines = [line.rstrip() for line in code.splitlines() if line.strip()]
        
        return "\n".join(lines)
    
    def _generate_diff(self, original: str, fixed: str) -> str:
        """Generate unified diff between original and fixed code."""
        original_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile='original',
            tofile='changed',
        )
        
        return ''.join(diff)
   
    def add_hypothesis(self, hypothesis: Dict, execution_result: Dict):
        """Add a new hypothesis with enhanced tracking."""
        self.total_attempts += 1
        
        fmethods = [hypothesis['methods'][m] for m in hypothesis.get('methods',[])]
        fmethodcodes = ''.join([self._remove_java_comments(m['fixed_method'])+'\n' for m in fmethods])

        # Create hypothesis record
        record = HypothesisRecord(
            id=self._generate_id(hypothesis.get('fixed_method', '')),
            hypothesis=hypothesis.get('hypothesis', ''),
            fixed_method=fmethodcodes,
            changes=hypothesis.get('changes', ''),
            approach_type=self._classify_approach(hypothesis),
            timestamp=time.time(),
            execution_result=execution_result,
            failure_reason=hypothesis.get('failure_reason'),
            parent_hypothesis_id=hypothesis.get('parent_id')
        )
        
        # Calculate metrics
        record.metrics = self._calculate_metrics(record)
        
        # Check for duplicates with similarity threshold
        if not self._is_duplicate_or_similar(record):
            self.hypotheses.append(record)
            
            # Update tracking
            self.tried_approaches.add(record.approach_type)
            if execution_result.get('status') == 'success':
                self.success_count += 1
                self.successful_patterns.append(record.approach_type)
            else:
                error_type = execution_result.get('error_type', 'Unknown')
                self.error_patterns[error_type] += 1
            
            # Maintain pool size with intelligent pruning
            if len(self.hypotheses) > self.max_size:
                self._prune_hypotheses()
            
            self.logger.info(f"Added hypothesis {record.id[:8]} with approach: {record.approach_type}")
    
    def _classify_approach(self, hypothesis: Dict) -> str:
        """Classify the approach type of a hypothesis."""

        changes = hypothesis.get('diff', '')
        ch=""
        if isinstance(changes, list):
            for change in changes:
                ch+=change
            changes = ch.lower()
        changes = changes.lower()

        hypothesis_text = hypothesis.get('hypothesis', '').lower()
        
        # Define approach patterns
        approach_patterns = {
            'null_check': ['null check', 'null pointer', '!= null', '== null'],
            'boundary_check': ['boundary', 'index', 'bounds', 'length check', 'size check'],
            'type_conversion': ['cast', 'conversion', 'type', 'instanceof'],
            'exception_handling': ['try', 'catch', 'exception', 'throw'],
            'initialization': ['initialize', 'init', 'default value', 'constructor'],
            'logic_fix': ['condition', 'if', 'else', 'logic', 'operator'],
            'loop_fix': ['loop', 'iteration', 'while', 'for'],
            'return_value': ['return', 'result'],
            'method_call': ['invoke', 'call', 'method'],
            'synchronization': ['synchronized', 'lock', 'thread', 'concurrent'],
            'validation': ['validate', 'verify', 'check', 'assert'],
            'refactoring': ['refactor', 'restructure', 'reorganize'],
        }
        
        combined_text = f"{changes} {hypothesis_text}"
        
        for approach, patterns in approach_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return approach
        
        return 'general_fix'
    
    def _calculate_metrics(self, record: HypothesisRecord) -> HypothesisMetrics:
        """Calculate metrics for hypothesis ranking."""
        metrics = HypothesisMetrics()
        
        # Complexity (based on change size and structure)
        metrics.complexity = self._calculate_complexity(record.fixed_method, record.changes)
        
        # Novelty (how different from previous attempts)
        metrics.novelty = self._calculate_novelty(record)
        
        # Confidence (based on approach success rate)
        metrics.confidence = self._calculate_confidence(record.approach_type)
        
        # Similarity to failed attempts
        metrics.similarity_to_failed = self._calculate_similarity_to_failed(record)
        
        # Approach diversity
        metrics.approach_diversity = self._calculate_approach_diversity(record.approach_type)
        
        # Error pattern match
        metrics.error_pattern_match = self._calculate_error_pattern_match(record)
        
        return metrics
    
    def _calculate_complexity(self, fixed_method: str, changes: str) -> float:
        """Calculate complexity score (0-100)."""
        score = 0.0
        
        # Length-based complexity
        score += min(len(fixed_method) / 50, 20)  # Cap at 20
        
        # Number of changes
        num_changes = changes.count('+') + changes.count('-')
        score += min(num_changes * 2, 30)  # Cap at 30
        
        # Structural complexity
        complexity_indicators = ['if', 'else', 'for', 'while', 'try', 'catch', 'switch']
        for indicator in complexity_indicators:
            score += fixed_method.count(indicator) * 2
        
        # Nesting depth (approximation)
        max_nesting = max(fixed_method.count('{'), fixed_method.count('('))
        score += min(max_nesting * 3, 20)
        
        return min(score, 100)  # Cap at 100
    
    def _calculate_novelty(self, record: HypothesisRecord) -> float:
        """Calculate how novel this hypothesis is (0-1)."""
        if len(self.hypotheses) < 2:
            return 1.0
        
        # Compare with recent hypotheses
        recent_hypotheses = self.hypotheses[-5:]
        
        # Check approach novelty
        approach_novelty = 1.0
        approach_count = sum(1 for h in recent_hypotheses if h.approach_type == record.approach_type)
        approach_novelty = 1.0 - (approach_count / len(recent_hypotheses))
        
        # Check code similarity
        code_novelty = 1.0
        for hyp in recent_hypotheses:
            similarity = self._calculate_code_similarity(record.fixed_method, hyp.fixed_method)
            code_novelty = min(code_novelty, 1.0 - similarity)
        
        return (approach_novelty * 0.6 + code_novelty * 0.4)
    
    def _calculate_confidence(self, approach_type: str) -> float:
        """Calculate confidence based on historical success rate (0-1)."""
        if self.total_attempts == 0:
            return 0.5  # Neutral confidence
        
        # Check success rate of this approach
        approach_attempts = sum(1 for h in self.hypotheses if h.approach_type == approach_type)
        approach_successes = sum(1 for h in self.hypotheses 
                                if h.approach_type == approach_type 
                                and h.execution_result.get('status') == 'success')
        
        if approach_attempts == 0:
            # New approach - give it moderate confidence
            return 0.6
        
        success_rate = approach_successes / approach_attempts
        
        # Adjust confidence based on sample size
        if approach_attempts < 3:
            # Small sample - regress toward mean
            return 0.5 + (success_rate - 0.5) * 0.5
        
        return success_rate
    
    def _calculate_similarity_to_failed(self, record: HypothesisRecord) -> float:
        """Calculate similarity to failed attempts (0-1)."""
        failed_hypotheses = [h for h in self.hypotheses 
                           if h.execution_result.get('status') == 'failed']
        
        if not failed_hypotheses:
            return 0.0
        
        max_similarity = 0.0
        for failed in failed_hypotheses[-5:]:  # Check recent failures
            # Compare approaches
            if record.approach_type == failed.approach_type:
                max_similarity = max(max_similarity, 0.5)
            
            # Compare code
            code_sim = self._calculate_code_similarity(record.fixed_method, failed.fixed_method)
            max_similarity = max(max_similarity, code_sim)
        
        return max_similarity
    
    def _calculate_approach_diversity(self, approach_type: str) -> float:
        """Calculate how this approach contributes to diversity (0-1)."""
        if not self.tried_approaches:
            return 1.0
        
        # Count recent uses of this approach
        recent_approaches = [h.approach_type for h in self.hypotheses[-5:]]
        if not recent_approaches:
            return 1.0
        
        approach_frequency = recent_approaches.count(approach_type) / len(recent_approaches)
        diversity_score = 1.0 - approach_frequency
        
        # Bonus for completely new approaches
        if approach_type not in self.tried_approaches:
            diversity_score = min(1.0, diversity_score + 0.3)
        
        return diversity_score
    
    def _calculate_error_pattern_match(self, record: HypothesisRecord) -> float:
        """Calculate how well this approach matches the error pattern (0-1)."""
        error_type = record.execution_result.get('error_type', 'Unknown')
        
        # Define approach-error affinity
        error_approach_affinity = {
            'NullPointerException': ['null_check', 'initialization', 'validation'],
            'IndexOutOfBoundsException': ['boundary_check', 'loop_fix', 'validation'],
            'ClassCastException': ['type_conversion', 'validation'],
            'AssertionError': ['logic_fix', 'return_value', 'validation'],
            'IllegalArgumentException': ['validation', 'initialization', 'boundary_check'],
            'ArithmeticException': ['validation', 'logic_fix'],
            'ConcurrentModificationException': ['synchronization', 'loop_fix'],
        }
        
        if error_type in error_approach_affinity:
            if record.approach_type in error_approach_affinity[error_type]:
                return 0.8
            return 0.3
        
        return 0.5  # Neutral for unknown error types
    
    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets (0-1)."""
        if not code1 or not code2:
            return 0.0
        
        # Normalize codes
        norm1 = self._normalize_code(code1)
        norm2 = self._normalize_code(code2)
        
        if norm1 == norm2:
            return 1.0
        
        # Token-based similarity
        tokens1 = set(re.findall(r'\w+', norm1))
        tokens2 = set(re.findall(r'\w+', norm2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        jaccard = len(intersection) / len(union)
        
        # Length similarity
        len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
        
        return jaccard * 0.7 + len_ratio * 0.3
    
    def _is_duplicate_or_similar(self, new_record: HypothesisRecord, threshold: float = 0.9) -> bool:
        """Check if hypothesis is duplicate or too similar to existing ones."""
        if not self.hypotheses:
            return False
        
        new_norm = self._normalize_code(new_record.fixed_method)
        
        for hyp in self.hypotheses:
            # Exact duplicate check
            if hyp.id == new_record.id:
                self.logger.debug(f"Exact duplicate found: {new_record.id[:8]}")
                return True
            
            # Normalized code check
            existing_norm = self._normalize_code(hyp.fixed_method)
            if new_norm == existing_norm:
                self.logger.debug("Normalized duplicate found")
                return True
            
            # # High similarity check
            # similarity = self._calculate_code_similarity(new_record.fixed_method, hyp.fixed_method)
            # if similarity >= threshold:
            #     self.logger.debug(f"High similarity ({similarity:.2f}) found with {hyp.id[:8]}")
            #     return True
        
        return False
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        if not code:
            return ""
        
        # Remove comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        # Remove spaces around operators
        code = re.sub(r'\s*([{}();,=<>!+\-*/])\s*', r'\1', code)
        
        return code.strip()
    
    def _generate_id(self, code: str) -> str:
        """Generate unique ID for hypothesis."""
        if not code:
            return hashlib.md5(str(time.time()).encode()).hexdigest()
        return hashlib.md5(code.encode()).hexdigest()
    
    def _prune_hypotheses(self):
        """Intelligently prune hypothesis pool to maintain size."""
        # Keep all successful hypotheses
        successful = [h for h in self.hypotheses if h.execution_result.get('status') == 'success']
        failed = [h for h in self.hypotheses if h.execution_result.get('status') == 'failed']
        
        # Sort failed by metrics
        failed.sort(key=lambda h: h.metrics.get_overall_score(), reverse=True)
        
        # Keep top failed hypotheses
        keep_failed = self.max_size - len(successful)
        self.hypotheses = successful + failed[:keep_failed]
        
        self.logger.debug(f"Pruned pool to {len(self.hypotheses)} hypotheses")
    
    def get_recent_hypotheses(self, n: int = 5) -> List[Dict]:
        """Get n most recent hypotheses as dictionaries."""
        if n <= 0:
            return []
        
        recent = self.hypotheses[-n:] if self.hypotheses else []
        
        # Convert to dictionaries for compatibility
        return [
            {
                'id': h.id,
                'hypothesis': h.hypothesis,
                'fixed_method': h.fixed_method,
                'changes': h.changes,
                'approach_type': h.approach_type,
                'execution_result': h.execution_result,
                'failure_reason': h.failure_reason,
                'metrics': {
                    'score': h.metrics.get_overall_score(),
                    'novelty': h.metrics.novelty,
                    'confidence': h.metrics.confidence
                }
            }
            for h in recent
        ]
    
    def get_failed_patterns(self) -> List[Dict]:
        """Extract patterns from failed hypotheses with detailed analysis."""
        patterns = defaultdict(lambda: {
            'error_type': '',
            'count': 0,
            'attempted_approaches': set(),
            'common_failures': [],
            'avg_complexity': 0.0,
            'suggestions': []
        })
        
        failed_hypotheses = [h for h in self.hypotheses if h.execution_result.get('status') == 'failed']
        
        for hyp in failed_hypotheses:
            error_type = hyp.execution_result.get('error_type', 'unknown')
            pattern = patterns[error_type]
            
            pattern['error_type'] = error_type
            pattern['count'] += 1
            pattern['attempted_approaches'].add(hyp.approach_type)
            
            if hyp.failure_reason and hyp.failure_reason not in pattern['common_failures']:
                pattern['common_failures'].append(hyp.failure_reason)
            
            pattern['avg_complexity'] += hyp.metrics.complexity
        
        # Calculate averages and generate suggestions
        for error_type, pattern in patterns.items():
            if pattern['count'] > 0:
                pattern['avg_complexity'] /= pattern['count']
                pattern['attempted_approaches'] = list(pattern['attempted_approaches'])
                pattern['suggestions'] = self._generate_suggestions(error_type, pattern)
        
        return list(patterns.values())
    
    def _generate_suggestions(self, error_type: str, pattern: Dict) -> List[str]:
        """Generate suggestions based on error patterns."""
        suggestions = []
        
        # Error-specific suggestions
        error_suggestions = {
            'NullPointerException': [
                "Add comprehensive null checks before object access",
                "Initialize objects in constructor or declaration",
                "Use Optional for nullable returns"
            ],
            'IndexOutOfBoundsException': [
                "Validate array/list indices before access",
                "Check collection size in loop conditions",
                "Use iterator instead of index-based access"
            ],
            'AssertionError': [
                "Review test expectations and method contract",
                "Check edge cases in logic",
                "Verify return value calculations"
            ],
            'ClassCastException': [
                "Add instanceof checks before casting",
                "Review generic type usage",
                "Use proper type conversions"
            ],
            'IllegalArgumentException': [
                "Validate method parameters at entry",
                "Add precondition checks",
                "Review method documentation for constraints"
            ]
        }
        
        if error_type in error_suggestions:
            # Filter out already tried approaches
            for suggestion in error_suggestions[error_type]:
                approach_keywords = suggestion.lower().split()
                if not any(approach in pattern['attempted_approaches'] for approach in approach_keywords):
                    suggestions.append(suggestion)
        
        # Generic suggestions based on patterns
        if pattern['count'] > 3:
            suggestions.append("Consider a fundamentally different approach")
        
        if pattern['avg_complexity'] > 50:
            suggestions.append("Try simpler, more targeted fixes")
        
        return suggestions
    
    def should_try_different_approach(self) -> bool:
        """Determine if we should try a fundamentally different approach."""
        recent = self.hypotheses[-5:] if len(self.hypotheses) >= 5 else self.hypotheses
        
        if len(recent) < 3:
            return False
        
        # Check for repeated errors
        recent_errors = [h.execution_result.get('error_type') for h in recent]
        if len(set(recent_errors)) == 1:
            self.logger.info("Same error repeating - suggesting different approach")
            return True
        
        # Check for repeated approaches
        recent_approaches = [h.approach_type for h in recent]
        if len(set(recent_approaches)) <= 2:
            self.logger.info("Limited approach diversity - suggesting different approach")
            return True
        
        # Check for declining scores
        recent_scores = [h.metrics.get_overall_score() for h in recent]
        if all(recent_scores[i] >= recent_scores[i+1] for i in range(len(recent_scores)-1)):
            self.logger.info("Declining hypothesis quality - suggesting different approach")
            return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the hypothesis pool."""
        if not self.hypotheses:
            return {'total': 0, 'message': 'No hypotheses generated yet'}
        
        successful = [h for h in self.hypotheses if h.execution_result.get('status') == 'success']
        failed = [h for h in self.hypotheses if h.execution_result.get('status') == 'failed']
        
        # Calculate approach distribution
        approach_dist = Counter(h.approach_type for h in self.hypotheses)
        
        # Calculate average metrics
        avg_metrics = {
            'complexity': np.mean([h.metrics.complexity for h in self.hypotheses]),
            'novelty': np.mean([h.metrics.novelty for h in self.hypotheses]),
            'confidence': np.mean([h.metrics.confidence for h in self.hypotheses])
        }
        
        return {
            'total': len(self.hypotheses),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.hypotheses) if self.hypotheses else 0,
            'error_distribution': dict(self.error_patterns),
            'approach_distribution': dict(approach_dist),
            'average_metrics': avg_metrics,
            'pool_capacity': f"{len(self.hypotheses)}/{self.max_size}",
            'diversity_score': len(self.tried_approaches) / max(len(self.hypotheses), 1),
            'should_change_approach': self.should_try_different_approach()
        }


class ApproachTaxonomy:
    """Taxonomy of fix approaches for classification."""
    
    def __init__(self):
        self.taxonomy = {
            'defensive': ['null_check', 'boundary_check', 'validation', 'exception_handling'],
            'structural': ['initialization', 'refactoring', 'synchronization'],
            'logical': ['logic_fix', 'loop_fix', 'return_value'],
            'type_related': ['type_conversion', 'method_call'],
        }
        
        self.inverse_taxonomy = {}
        for category, approaches in self.taxonomy.items():
            for approach in approaches:
                self.inverse_taxonomy[approach] = category
    
    def get_category(self, approach_type: str) -> str:
        """Get the category of an approach."""
        return self.inverse_taxonomy.get(approach_type, 'general')
    
    def get_related_approaches(self, approach_type: str) -> List[str]:
        """Get approaches related to the given one."""
        category = self.get_category(approach_type)
        if category in self.taxonomy:
            return [a for a in self.taxonomy[category] if a != approach_type]
        return []


class HypothesisRanker:
    """Rank and prioritize hypotheses for generation and testing."""
    
    def __init__(self, pool: HypothesisPool):
        self.pool = pool
        self.logger = logging.getLogger(__name__)
    
    def rank_hypotheses(self, candidates: List[Dict]) -> List[Dict]:
        """Rank candidate hypotheses based on multiple factors."""
        ranked = []
        
        for candidate in candidates:
            # Calculate score
            score = self._calculate_score(candidate)
            candidate['rank_score'] = score
            ranked.append(candidate)
        
        # Sort by score (highest first)
        ranked.sort(key=lambda x: x['rank_score'], reverse=True)
        
        self.logger.debug(f"Ranked {len(ranked)} hypotheses, top score: {ranked[0]['rank_score']:.3f}")
        
        return ranked
    
    def _calculate_score(self, candidate: Dict) -> float:
        """Calculate ranking score for a candidate hypothesis."""
        score = 0.0
        
        # Base score
        score = 50.0
        
        # Penalize repeated error types
        error_type = candidate.get('expected_error_type', 'Unknown')
        if error_type in self.pool.error_patterns:
            penalty = min(self.pool.error_patterns[error_type] * 5, 30)
            score -= penalty
        
        # Reward novel approaches
        approach = candidate.get('approach_type', 'general_fix')
        if approach not in self.pool.tried_approaches:
            score += 20
        else:
            # Penalize overused approaches
            approach_count = sum(1 for h in self.pool.hypotheses if h.approach_type == approach)
            score -= min(approach_count * 3, 15)
        
        # Consider complexity
        estimated_complexity = len(candidate.get('changes', '')) * 0.1
        score -= min(estimated_complexity, 10)
        
        # Reward if addresses specific error patterns
        if self._matches_error_pattern(candidate):
            score += 15
        
        # Historical success rate of similar approaches
        confidence = self._get_approach_confidence(approach)
        score += confidence * 20
        
        # Diversity bonus
        if self._increases_diversity(candidate):
            score += 10
        
        return max(score, 0)  # Ensure non-negative
    
    def _matches_error_pattern(self, candidate: Dict) -> bool:
        """Check if candidate matches known error patterns."""
        if not self.pool.error_patterns:
            return False
        
        # Get most common error
        most_common_error = self.pool.error_patterns.most_common(1)[0][0]
        
        error_approach_map = {
            'NullPointerException': ['null_check', 'initialization'],
            'IndexOutOfBoundsException': ['boundary_check', 'validation'],
            'AssertionError': ['logic_fix', 'return_value'],
        }
        
        approach = candidate.get('approach_type', '')
        if most_common_error in error_approach_map:
            return approach in error_approach_map[most_common_error]
        
        return False
    
    def _get_approach_confidence(self, approach_type: str) -> float:
        """Get confidence score for an approach based on history."""
        if not self.pool.hypotheses:
            return 0.5
        
        same_approach = [h for h in self.pool.hypotheses if h.approach_type == approach_type]
        if not same_approach:
            return 0.6  # Slight positive bias for new approaches
        
        successes = sum(1 for h in same_approach if h.execution_result.get('status') == 'success')
        return successes / len(same_approach)
    
    def _increases_diversity(self, candidate: Dict) -> bool:
        """Check if candidate increases approach diversity."""
        if len(self.pool.hypotheses) < 3:
            return True
        
        recent_approaches = [h.approach_type for h in self.pool.hypotheses[-3:]]
        return candidate.get('approach_type') not in recent_approaches
    
    def suggest_next_approaches(self, n: int = 3) -> List[str]:
        """Suggest the next approaches to try."""
        suggestions = []
        
        # Get taxonomy
        taxonomy = ApproachTaxonomy()
        
        # Analyze what hasn't been tried
        all_approaches = set(taxonomy.inverse_taxonomy.keys())
        untried = all_approaches - self.pool.tried_approaches
        
        # Prioritize based on error patterns
        if self.pool.error_patterns:
            most_common_error = self.pool.error_patterns.most_common(1)[0][0]
            
            error_priority = {
                'NullPointerException': ['null_check', 'initialization', 'validation'],
                'IndexOutOfBoundsException': ['boundary_check', 'loop_fix', 'validation'],
                'AssertionError': ['logic_fix', 'return_value', 'validation'],
                'ClassCastException': ['type_conversion', 'validation'],
            }
            
            if most_common_error in error_priority:
                for approach in error_priority[most_common_error]:
                    if approach in untried and approach not in suggestions:
                        suggestions.append(approach)
        
        # Add untried approaches
        for approach in untried:
            if approach not in suggestions:
                suggestions.append(approach)
                if len(suggestions) >= n:
                    break
        
        # If still need more, suggest variations of successful approaches
        if len(suggestions) < n:
            for hyp in self.pool.hypotheses:
                if hyp.execution_result.get('status') == 'success':
                    related = taxonomy.get_related_approaches(hyp.approach_type)
                    for approach in related:
                        if approach not in suggestions:
                            suggestions.append(approach)
                            if len(suggestions) >= n:
                                break
        
        return suggestions[:n]


# Keep the original simple classes for compatibility
class FailureAnalyzer:
    """Analyze why patches failed with support for multiple test failures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, hypothesis: Dict, execution_result: Dict) -> str:
        """
        Analyze failure and return detailed reason.
        Now handles multiple test failures from error_details.
        """
        # Check if we have detailed error information
        error_details = execution_result.get('error_details', {})
        
        if error_details:
            # Analyze multiple test failures
            return self._analyze_multiple_failures(error_details, execution_result)
        else:
            # Fallback to single error analysis (backward compatibility)
            error_type = execution_result.get('error_type', 'Unknown')
            error_msg = execution_result.get('error_message', '')
            return self._analyze_single_failure(error_type, error_msg, execution_result)
    
    def _analyze_multiple_failures(self, error_details: Dict, execution_result: Dict) -> str:
        """Analyze multiple test failures and create comprehensive summary."""
        error_summary = execution_result.get('error_summary', {})
        total_failures = error_summary.get('total_failures', len(error_details))
        
        if total_failures == 0:
            return "No test failures detected"
        
        # Analyze error patterns
        error_patterns = defaultdict(list)
        for test_name, error_info in error_details.items():
            error_type = error_info.get('error_type', 'Unknown')
            error_patterns[error_type].append(test_name)
        
        # Create failure reason summary
        if len(error_patterns) == 1:
            # All tests failed with same error type
            error_type = list(error_patterns.keys())[0]
            failed_tests = error_patterns[error_type]
            
            if len(failed_tests) == 1:
                # Single test failure
                test_name = failed_tests[0]
                error_info = error_details[test_name]
                return f"{error_type} in {test_name}: {error_info.get('error_message', 'Unknown')[:200]}"
            else:
                # Multiple tests, same error
                return f"{error_type} in {len(failed_tests)} tests: {', '.join(failed_tests[:3])}"
        else:
            # Different error types
            summary_parts = []
            for error_type, tests in list(error_patterns.items())[:3]:  # Top 3 error types
                summary_parts.append(f"{error_type}({len(tests)} tests)")
            
            return f"Multiple error types: {', '.join(summary_parts)}"
    
    def _analyze_single_failure(self, error_type: str, error_msg: str, execution_result: Dict) -> str:
        """Analyze single failure (backward compatibility)."""
        if error_type == 'CompilationError':
            return self._analyze_compilation_error(error_msg)
        elif error_type == 'TestFailure':
            return self._analyze_test_failure(error_msg, execution_result)
        elif error_type == 'NullPointerException':
            return self._analyze_null_pointer(error_msg)
        elif error_type == 'IndexOutOfBoundsException':
            return self._analyze_index_bounds(error_msg)
        elif error_type == 'ClassCastException':
            return self._analyze_class_cast(error_msg)
        elif error_type == 'IllegalArgumentException':
            return self._analyze_illegal_argument(error_msg)
        elif error_type == 'AssertionFailure':
            return self._analyze_assertion_failure(error_msg, execution_result)
        else:
            return f"{error_type}: {error_msg[:100]}"
    
    def _analyze_assertion_failure(self, error_msg: str, result: Dict) -> str:
        """Analyze assertion failures with detailed information."""
        # Check for detailed assertion info
        error_details = result.get('error_details', {})
        
        if error_details:
            # Find assertion failures
            assertion_failures = []
            for test, details in error_details.items():
                if details.get('assertion'):
                    assertion = details['assertion']
                    assertion_failures.append(
                        f"{test}: expected {assertion.get('expected')} but got {assertion.get('actual')}"
                    )
            
            if assertion_failures:
                return f"Assertion failures: {'; '.join(assertion_failures[:2])}"
        
        return f"Test assertion failed: {error_msg[:200]}"
    
    def _analyze_test_failure(self, error_msg: str, result: Dict) -> str:
        """Analyze test failure details."""
        failed_tests = result.get('failed_tests', [])
        if failed_tests:
            return f"Tests failed: {', '.join(failed_tests[:3])}"
        return f"Test assertion failed: {error_msg[:100]}"
    
    def _analyze_null_pointer(self, error_msg: str) -> str:
        """Analyze null pointer exception."""
        import re
        match = re.search(r'Cannot invoke "([^"]+)"', error_msg)
        if match:
            return f"Null reference when calling: {match.group(1)}"
        return f"Null pointer access: {error_msg[:100]}"
    
    def _analyze_index_bounds(self, error_msg: str) -> str:
        """Analyze index out of bounds."""
        import re
        match = re.search(r'Index[: ]+(\d+).*[Ss]ize[: ]+(\d+)', error_msg)
        if match:
            return f"Index {match.group(1)} exceeds size {match.group(2)}"
        return f"Array/List index out of bounds: {error_msg[:100]}"
    
    def _analyze_class_cast(self, error_msg: str) -> str:
        """Analyze class cast exception."""
        import re
        match = re.search(r'cannot be cast to (.+)', error_msg)
        if match:
            return f"Invalid cast to {match.group(1)}"
        return f"Class cast exception: {error_msg[:100]}"
    
    def _analyze_illegal_argument(self, error_msg: str) -> str:
        """Analyze illegal argument exception."""
        return f"Invalid argument: {error_msg[:100]}"
    
    def _analyze_compilation_error(self, error_msg: str) -> str:
        """Analyze compilation error."""
        if 'cannot find symbol' in error_msg:
            return "Compilation error: undefined symbol"
        elif 'incompatible types' in error_msg:
            return "Compilation error: type mismatch"
        elif 'unreachable statement' in error_msg:
            return "Compilation error: unreachable code"
        return f"Compilation error: {error_msg[:100]}"


class HypothesisUpdater:
    """Update hypothesis pool with analysis of multiple test failures."""
    
    def __init__(self):
        self.failure_analyzer = FailureAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    def analyze_and_update(self, 
                          hypothesis: Dict,
                          execution_result: Dict,
                          pool: 'HypothesisPool') -> Dict:
        """
        Analyze execution result with multiple test failures and update pool.
        """
        
        # Analyze failures (now handles multiple)
        failure_reason = self.failure_analyzer.analyze(
            hypothesis,
            execution_result
        )
        
        hypothesis['failure_reason'] = failure_reason
        
        # Add to pool
        pool.add_hypothesis(hypothesis, execution_result)
        
        # Generate insights with enhanced error analysis
        insights = self._generate_enhanced_insights(pool, execution_result, failure_reason)
        
        return insights
    
    def _generate_enhanced_insights(self, pool: 'HypothesisPool', 
                                   result: Dict, 
                                   failure_reason: str) -> Dict:
        """
        Generate insights with support for multiple error analysis.
        """
        
        # Get pool statistics
        stats = pool.get_statistics()
        
        # Base insights structure
        insights = {
            # Core fields
            'avoid_patterns': pool.get_failed_patterns(),
            # 'error_type': result.get('error_type', 'Unknown'),
            # 'error_message': result.get('error_message', ''),
            # 'failure_reason': failure_reason,
            'suggested_focus': self._suggest_focus_enhanced(result, pool),
            'suggested_approaches': self._suggest_approaches_enhanced(result, pool, stats),
            
            # NEW: Multiple error analysis
            'all_error_types': result.get('all_error_types', []),
            'error_details': result.get('error_details', {}),
            'error_summary': result.get('error_summary', {}),
            'per_test_analysis': self._analyze_per_test_failures(result),
            
            # Pool analysis
            'pool_statistics': stats,
            'should_change_approach': stats.get('should_change_approach', False),
            'no_new_patterns': False,
            'stuck_pattern': False,
            
            # Quality tracking
            'hypothesis_quality_trend': self._analyze_quality_trend(pool),
            'diversity_score': stats.get('diversity_score', 0),
            
            # Specific suggestions
            'specific_suggestions': [],
            'test_specific_suggestions': self._get_test_specific_suggestions(result),
            
            # Iteration tracking
            'total_attempts': stats.get('total_attempts', 0),
            'success_rate': stats.get('success_rate', 0),
            'recent_errors': []
        }
        
        # Check for patterns in multiple failures
        insights = self._analyze_failure_patterns(insights, result, pool)
        
        return insights
    
    def _analyze_per_test_failures(self, result: Dict) -> Dict:
        """Analyze failures on a per-test basis."""
        error_details = result.get('error_details', {})
        if not error_details:
            return {}
        
        analysis = {
            'total_failed_tests': len(error_details),
            'error_type_by_test': {},
            'common_error_locations': [],
            'test_groups': defaultdict(list)
        }
        
        # Group tests by error type
        for test_name, error_info in error_details.items():
            error_type = error_info.get('error_type', 'Unknown')
            analysis['error_type_by_test'][test_name] = error_type
            analysis['test_groups'][error_type].append(test_name)
        
        # Find common error locations
        error_lines = [info.get('error_line') for info in error_details.values() 
                      if info.get('error_line')]
        if error_lines:
            from collections import Counter
            line_counts = Counter(error_lines)
            analysis['common_error_locations'] = [
                line for line, count in line_counts.most_common(3) if count > 1
            ]
        
        return analysis
    
    def _suggest_focus_enhanced(self, result: Dict, pool: 'HypothesisPool') -> str:
        """Enhanced focus suggestion based on multiple errors."""
        error_details = result.get('error_details', {})
        error_summary = result.get('error_summary', {})
        
        if not error_details:
            # Fallback to single error analysis
            return self._suggest_focus_single(result.get('failure_reason', ''), pool)
        
        # Analyze error distribution
        error_distribution = error_summary.get('error_type_distribution', {})
        
        if not error_distribution:
            return "Analyze test failures individually"
        
        # Get dominant error type
        dominant_error = max(error_distribution.items(), key=lambda x: x[1])[0]
        count = error_distribution[dominant_error]
        total = sum(error_distribution.values())
        
        if count == total:
            # All tests fail with same error
            return self._get_focus_for_error_type(dominant_error)
        elif count / total > 0.7:
            # Majority same error
            return f"Primary focus: {self._get_focus_for_error_type(dominant_error)} ({count}/{total} tests)"
        else:
            # Mixed errors
            top_errors = sorted(error_distribution.items(), key=lambda x: x[1], reverse=True)[:2]
            focuses = [self._get_focus_for_error_type(err) for err, _ in top_errors]
            return f"Multiple issues: {' AND '.join(focuses)}"
    
    def _get_focus_for_error_type(self, error_type: str) -> str:
        """Get specific focus suggestion for an error type."""
        focus_map = {
            'NullPointerException': "null safety and object initialization",
            'IndexOutOfBoundsException': "array/collection bounds checking",
            'ArrayIndexOutOfBoundsException': "array bounds and length validation",
            'ClassCastException': "type compatibility and conversions",
            'IllegalArgumentException': "method parameter validation",
            'AssertionFailure': "expected behavior and test assertions",
            'CompilationError': "syntax and symbol definitions",
            'TestFailure': "test expectations and logic flow"
        }
        return focus_map.get(error_type, f"issues related to {error_type}")
    
    def _suggest_approaches_enhanced(self, result: Dict, pool: 'HypothesisPool', stats: Dict) -> List[str]:
        """Suggest approaches based on multiple error analysis."""
        suggestions = []
        
        # Get all error types
        all_errors = result.get('all_error_types', [result.get('error_type')])
        
        # Map errors to approaches
        error_approach_map = {
            'NullPointerException': ['null_check', 'initialization', 'defensive'],
            'IndexOutOfBoundsException': ['boundary_check', 'validation', 'loop_fix'],
            'AssertionFailure': ['logic_fix', 'condition_fix', 'state_fix'],
            'ClassCastException': ['type_conversion', 'cast_fix', 'generic_fix'],
            'IllegalArgumentException': ['validation', 'parameter_fix', 'precondition']
        }
        
        # Collect suggested approaches for all error types
        suggested_approaches = set()
        for error_type in all_errors:
            if error_type in error_approach_map:
                suggested_approaches.update(error_approach_map[error_type])
        
        # Filter out already tried approaches
        approach_dist = stats.get('approach_distribution', {})
        untried = [a for a in suggested_approaches if a not in approach_dist]
        less_tried = [a for a in suggested_approaches if approach_dist.get(a, 0) <= 1]
        
        # Prioritize untried, then less tried
        suggestions.extend(untried)
        suggestions.extend([a for a in less_tried if a not in untried])
        
        # Add generic suggestions if needed
        if len(suggestions) < 3:
            generic = ['refactoring', 'defensive_programming', 'error_handling']
            suggestions.extend([g for g in generic if g not in suggestions])
        
        return suggestions[:5]  # Return top 5
    
    def _get_test_specific_suggestions(self, result: Dict) -> Dict[str, str]:
        """Get suggestions specific to each failing test."""
        error_details = result.get('error_details', {})
        if not error_details:
            return {}
        
        suggestions = {}
        
        for test_name, error_info in error_details.items():
            error_type = error_info.get('error_type', 'Unknown')
            error_msg = error_info.get('error_message', '')
            
            # Generate specific suggestion for this test
            if error_type == 'NullPointerException':
                suggestions[test_name] = f"Add null check before accessing object in test {test_name}"
            elif error_type == 'IndexOutOfBoundsException':
                if 'Index' in error_msg and 'Size' in error_msg:
                    suggestions[test_name] = f"Check array/list bounds before access in {test_name}"
                else:
                    suggestions[test_name] = f"Validate collection size in {test_name}"
            elif error_type == 'AssertionFailure':
                assertion = error_info.get('assertion')
                if assertion:
                    suggestions[test_name] = f"Fix logic to return {assertion.get('expected')} instead of {assertion.get('actual')}"
                else:
                    suggestions[test_name] = f"Review assertion expectations in {test_name}"
            else:
                suggestions[test_name] = f"Address {error_type} in {test_name}"
        
        return suggestions
    
    def _analyze_failure_patterns(self, insights: Dict, result: Dict, pool: 'HypothesisPool') -> Dict:
        """Analyze patterns across multiple test failures."""
        error_details = result.get('error_details', {})
        
        if not error_details:
            return insights
        
        # Check if same error keeps appearing
        recent = pool.get_recent_hypotheses(3)
        if len(recent) >= 3:
            recent_error_details = []
            for hyp in recent:
                exec_result = hyp.get('execution_result', {})
                if 'error_details' in exec_result:
                    recent_error_details.append(exec_result['error_details'])
            
            # Check for stuck patterns
            if recent_error_details:
                # Count how many tests consistently fail
                consistent_failures = set()
                for test_name in error_details.keys():
                    if all(test_name in details for details in recent_error_details):
                        consistent_failures.add(test_name)
                
                if len(consistent_failures) > len(error_details) * 0.7:
                    insights['stuck_pattern'] = True
                    insights['consistently_failing_tests'] = list(consistent_failures)
                    insights['suggested_focus'] = f"Focus on fixing: {', '.join(list(consistent_failures)[:2])}"
        
        return insights
    
    def _suggest_focus_single(self, failure_reason: str, pool: 'HypothesisPool') -> str:
        """Original single-error focus suggestion for backward compatibility."""
        failure_lower = failure_reason.lower()
        
        if 'null' in failure_lower:
            return "Focus on null safety and object initialization"
        elif 'index' in failure_lower or 'bound' in failure_lower:
            return "Focus on array/collection bounds checking"
        elif 'cast' in failure_lower or 'type' in failure_lower:
            return "Focus on type compatibility and conversions"
        elif 'compilation' in failure_lower:
            stats = pool.get_statistics()
            if stats.get('total', 0) > 2:
                return "Review method signature and variable declarations"
            return "Check syntax and symbol definitions"
        elif 'test' in failure_lower or 'assert' in failure_lower:
            return "Analyze test expectations and assertions"
        else:
            stats = pool.get_statistics()
            if stats.get('should_change_approach'):
                return "Current approach not working - try different strategy"
            return "Refine current approach with small adjustments"
    
    def _analyze_quality_trend(self, pool: 'HypothesisPool') -> str:
        """Analyze the trend in hypothesis quality."""
        if len(pool.hypotheses) < 2:
            return "insufficient_data"
        
        # Get quality scores over time
        scores = [h.metrics.get_overall_score() for h in pool.hypotheses]
        
        # Calculate trend
        first_half_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
        second_half_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        
        if second_half_avg > first_half_avg * 1.1:
            return "improving"
        elif second_half_avg < first_half_avg * 0.9:
            return "declining"
        else:
            return "stable"

class HypothesisEvolution:
    """Track and manage hypothesis evolution over iterations."""
    
    def __init__(self):
        self.evolution_tree = {}
        self.successful_lineages = []
        self.logger = logging.getLogger(__name__)
    
    def track_evolution(self, parent_id: Optional[str], child: HypothesisRecord):
        """Track the evolution from parent to child hypothesis."""
        if parent_id:
            if parent_id not in self.evolution_tree:
                self.evolution_tree[parent_id] = []
            self.evolution_tree[parent_id].append(child.id)
            
            # Track successful lineages
            if child.execution_result.get('status') == 'success':
                lineage = self._get_lineage(child.id)
                self.successful_lineages.append(lineage)
    
    def _get_lineage(self, hypothesis_id: str) -> List[str]:
        """Get the complete lineage of a hypothesis."""
        lineage = [hypothesis_id]
        
        # Traverse back to find parents
        for parent_id, children in self.evolution_tree.items():
            if hypothesis_id in children:
                lineage = self._get_lineage(parent_id) + lineage
                break
        
        return lineage
    
    def get_successful_patterns(self) -> List[Dict]:
        """Extract patterns from successful hypothesis lineages."""
        patterns = []
        
        for lineage in self.successful_lineages:
            if len(lineage) > 1:
                patterns.append({
                    'lineage_length': len(lineage),
                    'evolution_path': lineage,
                    'final_success': lineage[-1]
                })
        
        return patterns
    
    def suggest_evolution(self, current: HypothesisRecord) -> Dict:
        """Suggest how to evolve a hypothesis based on patterns."""
        suggestions = {
            'refinement': None,
            'combination': None,
            'simplification': None
        }
        
        # Suggest refinement
        if current.metrics.confidence > 0.5:
            suggestions['refinement'] = "Refine current approach with minor adjustments"
        
        # Suggest combination
        if len(self.successful_lineages) > 0:
            suggestions['combination'] = "Combine with elements from successful approaches"
        
        # Suggest simplification
        if current.metrics.complexity > 70:
            suggestions['simplification'] = "Simplify by removing unnecessary changes"
        
        return suggestions