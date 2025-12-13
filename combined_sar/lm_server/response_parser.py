"""
Response parser and hallucination detector for LM server outputs.
Validates LLM responses and categorizes different types of hallucinations.
"""

import json
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum


class HallucinationType(Enum):
    """Types of hallucinations detected in LLM responses."""
    NONE = "none"
    INVALID_JSON = "invalid_json"
    MISSING_FIELDS = "missing_fields"
    INVALID_TOOL = "invalid_tool"
    INVALID_PARAMETERS = "invalid_parameters"
    REDUNDANT_SCANNING = "redundant_scanning"
    APPROACH_EMPTY_DIRECTION = "approach_empty_direction"
    EXPLORE_EXPLORED_AREA = "explore_explored_area"
    INEFFICIENT_ACTIONS = "inefficient_actions"
    CONFLICTING_ACTIONS = "conflicting_actions"
    NONCOMPLIANT_TOOLS = "noncompliant_tools"


class ResponseParser:
    """Parser and validator for LM server responses."""
    
    # Valid tools and their parameters (must match LM server TOOL_DEFINITIONS)
    VALID_TOOLS = {
        "explore": {
            "required_params": ["direction", "distance"],
            "valid_values": {
                "direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            }
        },
        "scan_surroundings": {
            "required_params": ["direction"],
            "valid_values": {
                "direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            },
            "optional": True  # Direction is optional for full 360° scan
        },
        "explore_obstacle": {
            "required_params": ["direction"],
            "valid_values": {
                "direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            }
        }
    }
    
    def __init__(self):
        self.hallucinations = []
    
    def parse_and_validate(
        self,
        response: Any,
        observations: Optional[Dict] = None
    ) -> Tuple[bool, List[Tuple[HallucinationType, str]]]:
        """
        Parse and validate LLM response, detecting hallucinations.
        
        Args:
            response: The raw response from LM server
            observations: Optional observation context for validation
            
        Returns:
            Tuple of (is_valid, list of (hallucination_type, description))
        """
        self.hallucinations = []
        
        # Check if response is valid JSON dict
        if not isinstance(response, dict):
            self.hallucinations.append((
                HallucinationType.INVALID_JSON,
                f"Response is not a dictionary: {type(response)}"
            ))
            return False, self.hallucinations
        
        # Check for required top-level fields
        if "reasoning" not in response:
            self.hallucinations.append((
                HallucinationType.MISSING_FIELDS,
                "Missing 'reasoning' field in response"
            ))
        
        if "actions" not in response:
            self.hallucinations.append((
                HallucinationType.MISSING_FIELDS,
                "Missing 'actions' field in response"
            ))
            return False, self.hallucinations
        
        # Validate actions list
        if not isinstance(response["actions"], list):
            self.hallucinations.append((
                HallucinationType.INVALID_JSON,
                f"'actions' must be a list, got {type(response['actions'])}"
            ))
            return False, self.hallucinations
        
        if len(response["actions"]) == 0:
            self.hallucinations.append((
                HallucinationType.MISSING_FIELDS,
                "No actions provided in response"
            ))
            return False, self.hallucinations
        
        # Validate each action
        for i, action in enumerate(response["actions"]):
            self._validate_action(action, i)
        
        # Context-aware validation if observations provided
        if observations and len(self.hallucinations) == 0:
            self._validate_with_context(response["actions"], observations)
        
        # Return result
        is_valid = len(self.hallucinations) == 0
        return is_valid, self.hallucinations
    
    def _validate_action(self, action: Any, index: int):
        """Validate a single action."""
        # Check if action is a dict
        if not isinstance(action, dict):
            self.hallucinations.append((
                HallucinationType.INVALID_JSON,
                f"Action {index} is not a dictionary: {type(action)}"
            ))
            return
        
        # Check for required fields
        if "action" not in action:
            self.hallucinations.append((
                HallucinationType.MISSING_FIELDS,
                f"Action {index} missing 'action' field"
            ))
            return
        
        if "parameters" not in action:
            self.hallucinations.append((
                HallucinationType.MISSING_FIELDS,
                f"Action {index} missing 'parameters' field"
            ))
            return
        
        tool_name = action["action"]
        parameters = action["parameters"]
        
        # Check if tool exists
        if tool_name not in self.VALID_TOOLS:
            self.hallucinations.append((
                HallucinationType.INVALID_TOOL,
                f"Action {index}: Unknown tool '{tool_name}'. Valid tools: {list(self.VALID_TOOLS.keys())}"
            ))
            return
        
        # Validate parameters
        tool_spec = self.VALID_TOOLS[tool_name]
        is_optional = tool_spec.get("optional", False)
        
        # Check required parameters (allow empty for optional tools like scan_surroundings)
        for param in tool_spec["required_params"]:
            if param not in parameters:
                # scan_surroundings with no direction means 360° scan
                if not (is_optional and tool_name == "scan_surroundings"):
                    self.hallucinations.append((
                        HallucinationType.INVALID_PARAMETERS,
                        f"Action {index} ({tool_name}): Missing required parameter '{param}'"
                    ))
                continue
            
            # Validate parameter values
            if param in tool_spec["valid_values"]:
                value = parameters[param]
                valid_values = tool_spec["valid_values"][param]
                
                # Convert distance to int for comparison if needed
                if param == "distance":
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        pass
                
                if value not in valid_values:
                    self.hallucinations.append((
                        HallucinationType.INVALID_PARAMETERS,
                        f"Action {index} ({tool_name}): Invalid {param}='{value}'. Valid: {valid_values}"
                    ))
    
    def _validate_with_context(self, actions: List[Dict], observations: Dict):
        """Validate actions against observation context."""
        # Parse observations
        drone_pos = observations.get("drone_position", {})
        map_info = observations.get("map", {})
        persons = observations.get("persons_detected", [])
        obstacles = observations.get("obstacles", [])
        
        # Get map coverage to determine if we should explore or scan
        coverage = map_info.get("coverage", 0.0)
        occupied_cells = map_info.get("occupied_cells", 0)
        
        # Count action types
        action_counts = {}
        scan_actions = []
        explore_obstacle_actions = []
        explore_actions = []
        
        for i, action in enumerate(actions):
            tool_name = action.get("action", "")
            params = action.get("parameters", {})
            direction = params.get("direction", "")
            
            # Count action types
            action_counts[tool_name] = action_counts.get(tool_name, 0) + 1
            
            if tool_name == "scan_surroundings":
                scan_actions.append((i, direction))
            elif tool_name == "explore_obstacle":
                explore_obstacle_actions.append((i, direction))
            elif tool_name == "explore":
                distance = params.get("distance", 10)
                explore_actions.append((i, direction, distance))
            

        
        # Check for excessive scanning when coverage is already high
        if coverage > 0.5 and action_counts.get("scan_surroundings", 0) > 2:
            self.hallucinations.append((
                HallucinationType.REDUNDANT_SCANNING,
                f"High map coverage ({coverage:.1%}) but planning {action_counts['scan_surroundings']} scans"
            ))
        
        # If obstacles exist, require at least one obstacle-focused scan
        if isinstance(obstacles, list) and len(obstacles) > 0:
            if len(explore_obstacle_actions) == 0:
                self.hallucinations.append((
                    HallucinationType.NONCOMPLIANT_TOOLS,
                    "Obstacles present but plan does not include 'explore_obstacle'"
                ))
        
        # Check for inefficient action patterns - too many scans, not enough exploration
        if len(actions) > 3:
            scan_ratio = action_counts.get("scan_surroundings", 0) / len(actions)
            explore_ratio = action_counts.get("explore", 0) / len(actions)
            
            if scan_ratio > 0.6 and coverage < 0.3:
                self.hallucinations.append((
                    HallucinationType.INEFFICIENT_ACTIONS,
                    f"Plan is {scan_ratio*100:.0f}% scanning with only {coverage:.1%} coverage - should explore more"
                ))
            
            if explore_ratio == 0 and occupied_cells < 100 and coverage < 0.4:
                self.hallucinations.append((
                    HallucinationType.INEFFICIENT_ACTIONS,
                    f"No exploration actions despite low coverage ({coverage:.1%})"
                ))
        
        # Check for conflicting consecutive actions
        for i in range(len(actions) - 1):
            current = actions[i]
            next_action = actions[i + 1]
            
            curr_tool = current.get("action", "")
            next_tool = next_action.get("action", "")
            curr_dir = current.get("parameters", {}).get("direction", "")
            next_dir = next_action.get("parameters", {}).get("direction", "")
            
            # Exploring in opposite directions consecutively
            if curr_tool == "explore" and next_tool == "explore":
                if self._are_opposite_directions(curr_dir, next_dir):
                    self.hallucinations.append((
                        HallucinationType.CONFLICTING_ACTIONS,
                        f"Actions {i},{i+1}: Exploring {curr_dir} then immediately exploring opposite {next_dir}"
                    ))
            
            # Scanning same direction multiple times
            if curr_tool == "scan_surroundings" and next_tool == "scan_surroundings":
                if curr_dir == next_dir and curr_dir:
                    self.hallucinations.append((
                        HallucinationType.REDUNDANT_SCANNING,
                        f"Actions {i},{i+1}: Scanning direction '{curr_dir}' multiple times consecutively"
                    ))
        
        # Check for unrealistic exploration distances
        for i, direction, distance in explore_actions:
            try:
                dist_val = int(distance)
                if dist_val <= 0 or dist_val >= 30:
                    self.hallucinations.append((
                        HallucinationType.INVALID_PARAMETERS,
                        f"Action {i}: Invalid distance {dist_val}m (must be integer < 30)"
                    ))
            except (ValueError, TypeError):
                self.hallucinations.append((
                    HallucinationType.INVALID_PARAMETERS,
                    f"Action {i}: Distance must be integer, got {distance}"
                ))
    
    def _are_opposite_directions(self, dir1: str, dir2: str) -> bool:
        """Check if two directions are opposite."""
        opposites = {
            "N": "S", "S": "N",
            "E": "W", "W": "E",
            "NE": "SW", "SW": "NE",
            "NW": "SE", "SE": "NW"
        }
        return opposites.get(dir1, "") == dir2
    
    def get_hallucination_summary(self) -> Dict[str, Any]:
        """Get summary of detected hallucinations."""
        if not self.hallucinations:
            return {
                "has_hallucinations": False,
                "count": 0,
                "types": []
            }
        
        # Count by type
        type_counts = {}
        for halluc_type, _ in self.hallucinations:
            type_name = halluc_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "has_hallucinations": True,
            "count": len(self.hallucinations),
            "types": list(type_counts.keys()),
            "type_counts": type_counts,
            "details": [
                {"type": h_type.value, "description": desc}
                for h_type, desc in self.hallucinations
            ]
        }
    
    def format_hallucination_report(self) -> str:
        """Format hallucinations as human-readable report."""
        if not self.hallucinations:
            return "✓ No hallucinations detected"
        
        report = [f"✗ Detected {len(self.hallucinations)} hallucination(s):\n"]
        
        for i, (halluc_type, description) in enumerate(self.hallucinations, 1):
            report.append(f"  {i}. [{halluc_type.value}] {description}")
        
        return "\n".join(report)


def validate_response(
    response: Any,
    observations: Optional[Dict] = None,
    print_report: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate a response.
    
    Args:
        response: LM server response
        observations: Optional observation context
        print_report: Whether to print the report
        
    Returns:
        Tuple of (is_valid, summary_dict)
    """
    parser = ResponseParser()
    is_valid, hallucinations = parser.parse_and_validate(response, observations)
    summary = parser.get_hallucination_summary()
    
    if print_report:
        print(parser.format_hallucination_report())
    
    return is_valid, summary
