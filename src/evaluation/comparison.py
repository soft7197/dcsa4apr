# src/evaluation/comparison.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

class ToolComparison:
    def __init__(self):
        self.tools = {
            'Our Approach': None,
            'TBar': None,
            'SimFix': None,
            'CapGen': None,
            'AVATAR': None,
            'FixMiner': None,
            'kPAR': None
        }
        
    def load_results(self, tool_name: str, results_file: str):
        """Load results for a specific tool."""
        with open(results_file, 'r') as f:
            self.tools[tool_name] = json.load(f)
    
    def compare_success_rates(self) -> pd.DataFrame:
        """Compare success rates across tools."""
        data = []
        
        for tool_name, results in self.tools.items():
            if results:
                data.append({
                    'Tool': tool_name,
                    'Plausible Patches': results.get('plausible_patches', 0),
                    'Correct Patches': results.get('correct_patches', 0),
                    'Success Rate': results.get('success_rate', 0)
                })
        
        return pd.DataFrame(data)
    
    def plot_comparison(self):
        """Generate comparison plots."""
        df = self.compare_success_rates()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Success rate comparison
        axes[0].bar(df['Tool'], df['Success Rate'])
        axes[0].set_title('Success Rate Comparison')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_xticklabels(df['Tool'], rotation=45)
        
        # Correct patches comparison
        axes[1].bar(df['Tool'], df['Correct Patches'])
        axes[1].set_title('Correct Patches Generated')
        axes[1].set_ylabel('Number of Correct Patches')
        axes[1].set_xticklabels(df['Tool'], rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/tool_comparison.png')
        plt.show()