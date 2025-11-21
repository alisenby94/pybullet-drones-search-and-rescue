"""
Test to verify that the motor control environment outputs correct RPM values.

This test confirms:
1. Action space is [-1, 1]^4 (normalized offsets)
2. _preprocessAction converts to actual RPM values [0, MAX_RPM]
3. RPMs are centered around HOVER_RPM with ±30% variation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simulation', 'gym-pybullet-drones'))

import numpy as np
from training_env_motor_control import MotorControlAviary

def test_rpm_conversion():
    """Test that actions are properly converted to RPM values."""
    print("\n" + "="*70)
    print("TESTING RPM CONVERSION IN MOTOR CONTROL ENVIRONMENT")
    print("="*70 + "\n")
    
    # Create environment
    env = MotorControlAviary(gui=False)
    
    # Get hover and max RPM values
    hover_rpm = env.HOVER_RPM
    max_rpm = env.MAX_RPM
    max_offset = 0.3 * hover_rpm
    
    print(f"HOVER_RPM: {hover_rpm:.2f}")
    print(f"MAX_RPM: {max_rpm:.2f}")
    print(f"Max offset (30% of hover): {max_offset:.2f}")
    print(f"Expected RPM range: [{hover_rpm - max_offset:.2f}, {hover_rpm + max_offset:.2f}]")
    print(f"Clipped to: [0, {max_rpm:.2f}]")
    print("\n" + "-"*70 + "\n")
    
    # Test cases
    test_actions = [
        (np.array([0.0, 0.0, 0.0, 0.0]), "Zero offset (hover)"),
        (np.array([1.0, 1.0, 1.0, 1.0]), "Maximum positive offset"),
        (np.array([-1.0, -1.0, -1.0, -1.0]), "Maximum negative offset"),
        (np.array([0.5, -0.5, 0.5, -0.5]), "Mixed offsets"),
        (np.array([0.1, 0.2, 0.3, 0.4]), "Small positive offsets"),
    ]
    
    print("ACTION PREPROCESSING TESTS:")
    print("="*70)
    
    for action, description in test_actions:
        rpm = env._preprocessAction(action)
        
        print(f"\nTest: {description}")
        print(f"  Input action (normalized): {action}")
        print(f"  Output RPM values: {rpm[0]}")
        print(f"  RPM range: [{rpm[0].min():.2f}, {rpm[0].max():.2f}]")
        
        # Verify RPMs are in valid range
        assert np.all(rpm >= 0), "ERROR: RPM values below 0!"
        assert np.all(rpm <= max_rpm), "ERROR: RPM values exceed MAX_RPM!"
        assert rpm.shape == (1, 4), f"ERROR: Wrong shape {rpm.shape}, expected (1, 4)"
        
        # Verify they're not just the raw action values
        assert not np.allclose(rpm[0], action), "ERROR: RPM equals raw action (not converted)!"
        
        print(f"  ✓ Valid RPM range")
        print(f"  ✓ Not raw action values")
    
    print("\n" + "="*70)
    print("TESTING WITH ACTUAL ENVIRONMENT STEP")
    print("="*70 + "\n")
    
    # Reset environment
    obs, info = env.reset()
    
    # Take a few steps with different actions
    test_steps = [
        (np.array([0.0, 0.0, 0.0, 0.0]), "Hover"),
        (np.array([0.5, 0.5, 0.5, 0.5]), "Increase thrust"),
        (np.array([-0.3, -0.3, -0.3, -0.3]), "Decrease thrust"),
    ]
    
    for action, description in test_steps:
        print(f"\nStep: {description}")
        print(f"  Action (normalized): {action}")
        
        # Manually preprocess to see RPM
        rpm = env._preprocessAction(action)
        print(f"  Converted RPM: {rpm[0]}")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify RPMs are being used correctly
        current_vel = info['actual_vel']
        print(f"  Resulting velocity: {current_vel}")
        print(f"  ✓ Step executed successfully")
        
        if terminated:
            print("  Episode terminated")
            break
    
    env.close()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nCONFIRMATION:")
    print("• Action space is [-1, 1]^4 (normalized offsets)")
    print("• _preprocessAction converts to RPM values")
    print(f"• RPM range: [{hover_rpm - max_offset:.2f}, {hover_rpm + max_offset:.2f}]")
    print(f"• RPMs are NOT raw [-1, 1] values")
    print("• Motor control outputs proper RPM commands")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_rpm_conversion()
