# src/utils/monitoring.py
import time
import psutil
import logging
from typing import Dict, Any
from datetime import datetime
import json

class PerformanceMonitor:
    def __init__(self, log_file: str = 'performance.log'):
        self.log_file = log_file
        self.metrics = []
        self.start_time = None
        
    def start_monitoring(self, bug_id: str):
        """Start monitoring for a bug fix attempt."""
        self.start_time = time.time()
        self.current_bug = bug_id
        self.iteration_metrics = []
        
    def log_iteration(self, iteration: int, metrics: Dict[str, Any]):
        """Log metrics for an iteration."""
        iteration_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'cpu_percent': psutil.cpu_percent(),
            **metrics
        }
        
        self.iteration_metrics.append(iteration_data)
    
    def end_monitoring(self, success: bool):
        """End monitoring and save results."""
        end_time = time.time()
        
        summary = {
            'bug_id': self.current_bug,
            'success': success,
            'total_time': end_time - self.start_time,
            'iterations': len(self.iteration_metrics),
            'iteration_details': self.iteration_metrics
        }
        
        self.metrics.append(summary)
        
        # Save to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
    
    def generate_report(self) -> Dict:
        """Generate performance report."""
        if not self.metrics:
            return {}
        
        total_bugs = len(self.metrics)
        successful = sum(1 for m in self.metrics if m['success'])
        
        avg_time = sum(m['total_time'] for m in self.metrics) / total_bugs
        avg_iterations = sum(m['iterations'] for m in self.metrics) / total_bugs
        
        return {
            'total_bugs_processed': total_bugs,
            'successful_fixes': successful,
            'average_time_seconds': avg_time,
            'average_iterations': avg_iterations,
            'success_rate': successful / total_bugs if total_bugs > 0 else 0
        }

class ErrorTracker:
    def __init__(self):
        self.errors = []
        
    def log_error(self, bug_id: str, error_type: str, error_msg: str, context: Dict = None):
        """Log an error occurrence."""
        error_entry = {
            'bug_id': bug_id,
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_msg,
            'context': context or {}
        }
        
        self.errors.append(error_entry)
        
        # Also log to standard logger
        logging.error(f"Bug {bug_id}: {error_type} - {error_msg}")
    
    def get_error_statistics(self) -> Dict:
        """Get statistics about errors."""
        if not self.errors:
            return {}
        
        error_types = {}
        for error in self.errors:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.errors),
            'error_types': error_types,
            'most_common_error': max(error_types, key=error_types.get)
        }