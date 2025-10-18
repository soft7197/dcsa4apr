# src/utils/trace_analyzer.py
"""Analyze and visualize bug fixing traces."""

import json
import os
import time
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class TraceAnalyzer:
    """Analyze bug fixing traces for insights."""
    
    def __init__(self, trace_dir: str = "traces"):
        self.trace_dir = trace_dir
        self.traces = {}
        self._load_traces()
    
    def _load_traces(self):
        """Load all trace files from directory."""
        trace_path = Path(self.trace_dir)
        if not trace_path.exists():
            return
        
        for trace_file in trace_path.glob("*_trace.json"):
            bug_id = trace_file.stem.replace("_trace", "")
            try:
                with open(trace_file, 'r') as f:
                    self.traces[bug_id] = json.load(f)
            except Exception as e:
                print(f"Error loading {trace_file}: {e}")
    
    def analyze_bug(self, bug_id: str) -> Dict:
        """Analyze a specific bug's trace."""
        if bug_id not in self.traces:
            return {"error": f"No trace found for {bug_id}"}
        
        trace = self.traces[bug_id]
        
        analysis = {
            'bug_id': bug_id,
            'success': trace.get('successful', False),
            'total_iterations': trace.get('total_iterations', 0),
            'total_time': trace.get('total_time', 0),
            'error_distribution': trace.get('error_patterns', {}),
            'tool_usage': self._analyze_tool_usage(trace),
            'hypothesis_patterns': self._analyze_hypothesis_patterns(trace),
            'context_efficiency': self._analyze_context_efficiency(trace),
            'convergence_rate': self._calculate_convergence_rate(trace)
        }
        
        return analysis
    
    def _analyze_tool_usage(self, trace: Dict) -> Dict:
        """Analyze tool usage patterns."""
        tool_usage = trace.get('tool_usage', [])
        if not tool_usage:
            return {}
        
        # Count by tool
        tool_counts = {}
        tool_timings = {}
        
        for usage in tool_usage:
            tool = usage['tool']
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            
            if 'timestamp' in usage:
                if tool not in tool_timings:
                    tool_timings[tool] = []
                tool_timings[tool].append(usage['timestamp'])
        
        # Calculate intervals
        tool_intervals = {}
        for tool, timestamps in tool_timings.items():
            if len(timestamps) > 1:
                intervals = [timestamps[i+1] - timestamps[i] 
                           for i in range(len(timestamps)-1)]
                tool_intervals[tool] = sum(intervals) / len(intervals)
        
        return {
            'counts': tool_counts,
            'average_intervals': tool_intervals,
            'most_used': max(tool_counts.items(), key=lambda x: x[1])[0] if tool_counts else None
        }
    
    def _analyze_hypothesis_patterns(self, trace: Dict) -> Dict:
        """Analyze hypothesis generation patterns."""
        hypotheses = trace.get('hypothesis_evolution', [])
        if not hypotheses:
            return {}
        
        # Approach distribution
        approaches = {}
        for hyp in hypotheses:
            approach = hyp.get('approach_type', 'unknown')
            approaches[approach] = approaches.get(approach, 0) + 1
        
        # Quality metrics over time
        complexities = [h.get('complexity', 0) for h in hypotheses]
        novelties = [h.get('novelty', 0) for h in hypotheses]
        confidences = [h.get('confidence', 0) for h in hypotheses]
        
        return {
            'approach_distribution': approaches,
            'average_complexity': sum(complexities) / len(complexities) if complexities else 0,
            'average_novelty': sum(novelties) / len(novelties) if novelties else 0,
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'quality_trend': self._detect_trend(confidences)
        }
    
    def _analyze_context_efficiency(self, trace: Dict) -> Dict:
        """Analyze context management efficiency."""
        context_evolution = trace.get('context_evolution', [])
        if not context_evolution:
            return {}
        
        sizes = [c.get('context_size_tokens', 0) for c in context_evolution]
        
        return {
            'initial_size': sizes[0] if sizes else 0,
            'final_size': sizes[-1] if sizes else 0,
            'max_size': max(sizes) if sizes else 0,
            'growth_rate': (sizes[-1] - sizes[0]) / len(sizes) if len(sizes) > 1 else 0,
            'efficiency': self._calculate_context_efficiency(trace)
        }
    
    def _calculate_context_efficiency(self, trace: Dict) -> float:
        """Calculate how efficiently context was used."""
        if not trace.get('successful'):
            return 0.0
        
        iterations = trace.get('total_iterations', 1)
        context_evolution = trace.get('context_evolution', [])
        
        if not context_evolution:
            return 0.0
        
        # Efficiency = success with minimal context growth
        avg_size = sum(c.get('context_size_tokens', 0) for c in context_evolution) / len(context_evolution)
        
        # Normalize (lower is better)
        efficiency = 1.0 / (1.0 + (avg_size / 1000) * (iterations / 10))
        
        return min(max(efficiency, 0.0), 1.0)
    
    def _calculate_convergence_rate(self, trace: Dict) -> Optional[float]:
        """Calculate how quickly the system converged to a solution."""
        if not trace.get('successful'):
            return None
        
        iterations = trace.get('total_iterations', 0)
        if iterations == 0:
            return None
        
        # Convergence rate: 1/iterations (normalized)
        return 1.0 / iterations
    
    def _detect_trend(self, values: List[float]) -> str:
        """Detect trend in a series of values."""
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple trend detection
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if second_half > first_half * 1.1:
            return "improving"
        elif second_half < first_half * 0.9:
            return "declining"
        else:
            return "stable"
    
    def compare_bugs(self, bug_ids: List[str] = None) -> pd.DataFrame:
        """Compare multiple bugs."""
        if bug_ids is None:
            bug_ids = list(self.traces.keys())
        
        comparisons = []
        for bug_id in bug_ids:
            if bug_id in self.traces:
                analysis = self.analyze_bug(bug_id)
                comparisons.append({
                    'Bug ID': bug_id,
                    'Success': analysis['success'],
                    'Iterations': analysis['total_iterations'],
                    'Time (s)': round(analysis['total_time'], 2),
                    'Convergence': analysis.get('convergence_rate', 0),
                    'Context Efficiency': analysis.get('context_efficiency', {}).get('efficiency', 0)
                })
        
        return pd.DataFrame(comparisons)
    
    def visualize_bug_trace(self, bug_id: str):
        """Create visualizations for a bug trace."""
        if bug_id not in self.traces:
            print(f"No trace found for {bug_id}")
            return
        
        trace = self.traces[bug_id]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Bug {bug_id} Analysis", fontsize=16)
        
        # 1. Error distribution
        ax1 = axes[0, 0]
        error_patterns = trace.get('error_patterns', {})
        if error_patterns:
            ax1.bar(error_patterns.keys(), error_patterns.values())
            ax1.set_title("Error Distribution")
            ax1.set_xlabel("Error Type")
            ax1.set_ylabel("Count")
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Context growth
        ax2 = axes[0, 1]
        context_evolution = trace.get('context_evolution', [])
        if context_evolution:
            iterations = [c.get('iteration', i) for i, c in enumerate(context_evolution)]
            sizes = [c.get('context_size_tokens', 0) for c in context_evolution]
            ax2.plot(iterations, sizes, marker='o')
            ax2.set_title("Context Size Evolution")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Tokens")
            ax2.grid(True, alpha=0.3)
        
        # 3. Hypothesis quality trends
        ax3 = axes[1, 0]
        hypothesis_evolution = trace.get('hypothesis_evolution', [])
        if hypothesis_evolution:
            iterations = list(range(1, len(hypothesis_evolution) + 1))
            confidences = [h.get('confidence', 0) for h in hypothesis_evolution]
            novelties = [h.get('novelty', 0) for h in hypothesis_evolution]
            
            ax3.plot(iterations, confidences, label='Confidence', marker='o')
            ax3.plot(iterations, novelties, label='Novelty', marker='s')
            ax3.set_title("Hypothesis Quality Metrics")
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Score")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Tool usage
        ax4 = axes[1, 1]
        tool_usage = {}
        for usage in trace.get('tool_usage', []):
            tool = usage['tool']
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        if tool_usage:
            ax4.pie(tool_usage.values(), labels=tool_usage.keys(), autopct='%1.1f%%')
            ax4.set_title("Tool Usage Distribution")
        
        plt.tight_layout()
        plt.savefig(f"traces/{bug_id}_visualization.png", dpi=150)
        plt.show()
    
    def generate_report(self, output_file: str = "traces/analysis_report.md"):
        """Generate comprehensive analysis report."""
        report = []
        report.append("# Bug Fixing Analysis Report\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall statistics
        report.append("## Overall Statistics\n")
        total_bugs = len(self.traces)
        successful = sum(1 for t in self.traces.values() if t.get('successful'))
        
        report.append(f"- Total bugs analyzed: {total_bugs}\n")
        report.append(f"- Successful fixes: {successful} ({100*successful/total_bugs:.1f}%)\n")
        
        if self.traces:
            avg_iterations = sum(t.get('total_iterations', 0) for t in self.traces.values()) / len(self.traces)
            avg_time = sum(t.get('total_time', 0) for t in self.traces.values()) / len(self.traces)
            
            report.append(f"- Average iterations: {avg_iterations:.1f}\n")
            report.append(f"- Average time: {avg_time:.1f} seconds\n")
        
        # Tool usage analysis
        report.append("\n## Tool Usage Analysis\n")
        all_tool_usage = {}
        for trace in self.traces.values():
            for usage in trace.get('tool_usage', []):
                tool = usage['tool']
                all_tool_usage[tool] = all_tool_usage.get(tool, 0) + 1
        
        if all_tool_usage:
            report.append("| Tool | Usage Count | Percentage |\n")
            report.append("|------|-------------|------------|\n")
            total_usage = sum(all_tool_usage.values())
            for tool, count in sorted(all_tool_usage.items(), key=lambda x: x[1], reverse=True):
                report.append(f"| {tool} | {count} | {100*count/total_usage:.1f}% |\n")
        
        # Error patterns
        report.append("\n## Common Error Patterns\n")
        all_errors = {}
        for trace in self.traces.values():
            for error, count in trace.get('error_patterns', {}).items():
                all_errors[error] = all_errors.get(error, 0) + count
        
        if all_errors:
            report.append("| Error Type | Occurrences |\n")
            report.append("|------------|-------------|\n")
            for error, count in sorted(all_errors.items(), key=lambda x: x[1], reverse=True):
                report.append(f"| {error} | {count} |\n")
        
        # Individual bug analysis
        report.append("\n## Individual Bug Analysis\n")
        
        comparison_df = self.compare_bugs()
        if not comparison_df.empty:
            report.append("\n")
            report.append(comparison_df.to_markdown(index=False))
        
        # Best and worst performers
        if self.traces:
            report.append("\n## Performance Highlights\n")
            
            # Fastest success
            successful_traces = [(bid, t) for bid, t in self.traces.items() if t.get('successful')]
            if successful_traces:
                fastest = min(successful_traces, key=lambda x: x[1].get('total_iterations', float('inf')))
                report.append(f"- **Fastest fix**: {fastest[0]} ({fastest[1].get('total_iterations')} iterations)\n")
                
                most_efficient = max(successful_traces, 
                                   key=lambda x: self._calculate_context_efficiency(x[1]))
                report.append(f"- **Most efficient**: {most_efficient[0]}\n")
            
            # Most challenging
            failed_traces = [(bid, t) for bid, t in self.traces.items() if not t.get('successful')]
            if failed_traces:
                most_attempts = max(failed_traces, key=lambda x: x[1].get('total_iterations', 0))
                report.append(f"- **Most challenging**: {most_attempts[0]} ({most_attempts[1].get('total_iterations')} attempts)\n")
        
        # Save report
        with open(output_file, 'w') as f:
            f.writelines(report)
        
        print(f"Report saved to {output_file}")
        return ''.join(report)
    
    def export_metrics_csv(self, output_file: str = "traces/metrics.csv"):
        """Export metrics to CSV for further analysis."""
        metrics = []
        
        for bug_id, trace in self.traces.items():
            analysis = self.analyze_bug(bug_id)
            
            metric_row = {
                'bug_id': bug_id,
                'success': trace.get('successful', False),
                'iterations': trace.get('total_iterations', 0),
                'time_seconds': trace.get('total_time', 0),
                'convergence_rate': analysis.get('convergence_rate', 0),
                'context_efficiency': analysis.get('context_efficiency', {}).get('efficiency', 0),
                'avg_hypothesis_complexity': analysis.get('hypothesis_patterns', {}).get('average_complexity', 0),
                'avg_hypothesis_novelty': analysis.get('hypothesis_patterns', {}).get('average_novelty', 0),
                'avg_hypothesis_confidence': analysis.get('hypothesis_patterns', {}).get('average_confidence', 0),
                'unique_approaches': len(analysis.get('hypothesis_patterns', {}).get('approach_distribution', {})),
                'unique_errors': len(trace.get('error_patterns', {})),
                'tools_used': len(set(u['tool'] for u in trace.get('tool_usage', [])))
            }
            
            metrics.append(metric_row)
        
        df = pd.DataFrame(metrics)
        df.to_csv(output_file, index=False)
        print(f"Metrics exported to {output_file}")
        return df


class RealTimeMonitor:
    """Real-time monitoring during bug fixing."""
    
    def __init__(self, bug_id: str):
        self.bug_id = bug_id
        self.start_time = time.time()
        self.current_iteration = 0
        
        # Setup live display
        self._setup_display()
    
    def _setup_display(self):
        """Setup live monitoring display."""
        print(f"\n{'='*60}")
        print(f"  Real-Time Monitor: {self.bug_id}")
        print(f"{'='*60}")
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")
    
    def update_iteration(self, iteration: int, hypothesis: str, approach: str):
        """Update display with iteration info."""
        self.current_iteration = iteration
        elapsed = time.time() - self.start_time
        
        print(f"[{elapsed:6.1f}s] Iteration {iteration:2d} | Approach: {approach:15s}")
        print(f"          Hypothesis: {hypothesis[:60]}...")
    
    def update_result(self, success: bool, error_type: str = None):
        """Update display with result."""
        if success:
            print(f"          ✓ SUCCESS!")
        else:
            print(f"          ✗ Failed: {error_type}")
    
    def update_tool_usage(self, tool: str, params: str):
        """Update display with tool usage."""
        print(f"          🔧 Tool: {tool} - {params[:40]}...")
    
    def finish(self, success: bool):
        """Finish monitoring."""
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*60}")
        if success:
            print(f"  ✓ Bug Fixed Successfully!")
        else:
            print(f"  ✗ Failed to Fix Bug")
        print(f"  Total Time: {elapsed:.1f}s | Iterations: {self.current_iteration}")
        print(f"{'='*60}\n")


# Utility functions for trace analysis

def analyze_trace_file(trace_file: str) -> Dict:
    """Quick analysis of a single trace file."""
    with open(trace_file, 'r') as f:
        trace = json.load(f)
    
    analyzer = TraceAnalyzer()
    analyzer.traces = {trace['bug_id']: trace}
    
    return analyzer.analyze_bug(trace['bug_id'])


def compare_trace_files(trace_files: List[str]) -> pd.DataFrame:
    """Compare multiple trace files."""
    analyzer = TraceAnalyzer()
    
    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            trace = json.load(f)
            analyzer.traces[trace['bug_id']] = trace
    
    return analyzer.compare_bugs()


def find_successful_patterns(trace_dir: str = "traces") -> Dict:
    """Find patterns in successful bug fixes."""
    analyzer = TraceAnalyzer(trace_dir)
    
    successful_traces = {
        bid: trace for bid, trace in analyzer.traces.items()
        if trace.get('successful')
    }
    
    if not successful_traces:
        return {"message": "No successful fixes found"}
    
    patterns = {
        'common_approaches': {},
        'effective_tools': {},
        'avg_iterations': 0,
        'avg_time': 0
    }
    
    for trace in successful_traces.values():
        # Collect approaches
        for hyp in trace.get('hypothesis_evolution', []):
            approach = hyp.get('approach_type', 'unknown')
            patterns['common_approaches'][approach] = patterns['common_approaches'].get(approach, 0) + 1
        
        # Collect tools
        for usage in trace.get('tool_usage', []):
            tool = usage['tool']
            patterns['effective_tools'][tool] = patterns['effective_tools'].get(tool, 0) + 1
        
        patterns['avg_iterations'] += trace.get('total_iterations', 0)
        patterns['avg_time'] += trace.get('total_time', 0)
    
    num_successful = len(successful_traces)
    patterns['avg_iterations'] /= num_successful
    patterns['avg_time'] /= num_successful
    
    return patterns


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            # Analyze all traces
            analyzer = TraceAnalyzer()
            report = analyzer.generate_report()
            print(report)
            
            # Export metrics
            analyzer.export_metrics_csv()
            
        elif sys.argv[1] == "visualize" and len(sys.argv) > 2:
            # Visualize specific bug
            bug_id = sys.argv[2]
            analyzer = TraceAnalyzer()
            analyzer.visualize_bug_trace(bug_id)
            
        elif sys.argv[1] == "patterns":
            # Find successful patterns
            patterns = find_successful_patterns()
            print(json.dumps(patterns, indent=2))
    else:
        print("Usage:")
        print("  python trace_analyzer.py analyze        # Generate analysis report")
        print("  python trace_analyzer.py visualize BUG  # Visualize specific bug")
        print("  python trace_analyzer.py patterns       # Find successful patterns")