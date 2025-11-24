"""
Verify Drone Exclusion Zone

Check that all obstacles are at least 2.0m away from drone spawn point.
"""

import numpy as np
from src.envs.action_coordinator_env import ActionCoordinatorEnv


def test_exclusion_zone():
    print("="*70)
    print("DRONE EXCLUSION ZONE TEST")
    print("="*70)
    
    # Test multiple resets to ensure exclusion zone always works
    num_tests = 5
    all_passed = True
    
    env = ActionCoordinatorEnv(
        gui=False,
        enable_obstacles=True,
        num_obstacles=20  # More obstacles = harder test
    )
    
    for test_num in range(num_tests):
        obs, info = env.reset()
        
        # Get drone position
        drone_state = env._getDroneStateVector(0)
        drone_pos = drone_state[0:3]
        
        print(f"\n📍 Test {test_num + 1}/{num_tests}")
        print(f"   Drone spawn: {drone_pos}")
        
        # Check all obstacles
        obstacle_info = env.obstacle_generator.get_obstacle_info()
        min_distance = float('inf')
        closest_obstacle = None
        
        violations = []
        for i, obs_info in enumerate(obstacle_info):
            obs_pos = obs_info['position']
            distance = np.linalg.norm(obs_pos - drone_pos)
            
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obs_info
            
            if distance < 2.0:
                violations.append((i, obs_info, distance))
        
        # Report results
        if violations:
            print(f"   ❌ FAILED: {len(violations)} obstacle(s) too close!")
            for i, obs_info, dist in violations:
                print(f"      [{i+1}] {obs_info['type']} at {obs_info['position']}: {dist:.2f}m")
            all_passed = False
        else:
            print(f"   ✅ PASSED: All obstacles respect 2.0m exclusion zone")
            print(f"      Closest: {closest_obstacle['type']} at {min_distance:.2f}m")
    
    env.close()
    
    print(f"\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Exclusion zone working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Check exclusion zone implementation")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = test_exclusion_zone()
    exit(0 if success else 1)
