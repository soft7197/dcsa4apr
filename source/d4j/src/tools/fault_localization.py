# src/tools/fault_localization.py
from typing import List, Dict, Tuple
import subprocess
import json
from dataclasses import dataclass
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FaultLocation:
    file_path: str
    class_name: str
    method_name: str
    suspiciousness_score: float
    line_numbers: List[int]
    
    # Additional fields for perfect FL
    buggy_code: Optional[str] = None
    fixed_code: Optional[str] = None
    buggy_fl: Optional[str] = None  # Buggy code with FL comment
    comment: Optional[str] = None
    similar_methods: Optional[List] = None
    normalized_body: Optional[List[str]] = None
    issue_title: Optional[str] = None
    issue_description: Optional[str] = None

class FaultLocalizer:
    def __init__(self, fl_tool="gzoltar", project_path: str = None):
        self.fl_tool = fl_tool
        self.project_path = project_path
        
    def run_fl_tool(self, failing_tests: List[str]) -> List[FaultLocation]:
        """Run fault localization tool (GZoltar for Java, coverage.py for Python)."""
        if self.fl_tool == "gzoltar":
            return self._run_gzoltar(failing_tests)
        elif self.fl_tool == "coverage":
            return self._run_coverage_py(failing_tests)
        else:
            # For perfect FL scenario
            return self._use_perfect_fl()
    
    def _run_gzoltar(self, failing_tests: List[str]) -> List[FaultLocation]:
        """Run GZoltar for Java projects."""
        cmd = [
            "java", "-jar", "lib/gzoltar.jar",
            "--projectDir", self.project_path,
            "--testMethods", ",".join(failing_tests),
            "--formula", "ochiai"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return self._parse_gzoltar_output(result.stdout)
    
    def _parse_gzoltar_output(self, output: str) -> List[FaultLocation]:
        """Parse GZoltar output to extract fault locations."""
        locations = []
        for line in output.split('\n'):
            if line.strip():
                parts = line.split(',')
                locations.append(FaultLocation(
                    file_path=parts[0],
                    class_name=parts[1],
                    method_name=parts[2],
                    suspiciousness_score=float(parts[3]),
                    line_numbers=[int(parts[4])]
                ))
        return sorted(locations, key=lambda x: x.suspiciousness_score, reverse=True)
    
    def find_connected_components(self, fault_locations: List[FaultLocation], 
                                 test_coverage: Dict) -> List[List[Tuple[FaultLocation, List[str]]]]:
        """Group fault locations and failing tests based on coverage."""
        components = []
        visited_methods = set()
        visited_tests = set()
        
        for fl in fault_locations:
            if fl.method_name not in visited_methods:
                component = []
                # Find all tests that cover this method
                covering_tests = test_coverage.get(fl.method_name, [])
                
                for test in covering_tests:
                    if test not in visited_tests:
                        component.append((fl, test))
                        visited_tests.add(test)
                
                if component:
                    components.append(component)
                    visited_methods.add(fl.method_name)
        
        return components