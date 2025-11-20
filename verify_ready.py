#!/usr/bin/env python3
"""
Quick verification script to check if the hierarchical system is ready to train.
"""

import os
import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - FILE MISSING: {path}")
        return False

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description} - IMPORT ERROR: {e}")
        return False

def main():
    print("=" * 70)
    print("HIERARCHICAL RL SYSTEM - READINESS CHECK")
    print("=" * 70)
    
    all_good = True
    
    # Check core environment files
    print("\n📁 ENVIRONMENT FILES:")
    all_good &= check_file("training_env_motor_control.py", "Stage 2 Motor Control Environment")
    all_good &= check_file("training_env_path_planning.py", "Stage 1 Path Planning Environment")
    
    # Check vision system
    print("\n👁️  VISION SYSTEM:")
    all_good &= check_file("stereo_vision.py", "Stereo Vision Module")
    all_good &= check_file("depth_downsampler.py", "Depth Downsampler")
    
    # Check training scripts
    print("\n🎓 TRAINING SCRIPTS:")
    all_good &= check_file("train_motor_control.py", "Stage 2 Training Script")
    
    # Check documentation
    print("\n📚 DOCUMENTATION:")
    all_good &= check_file("ARCHITECTURE.md", "Architecture Documentation")
    all_good &= check_file("TRAINING_STRATEGY.md", "Training Strategy Guide")
    
    # Check dependencies
    print("\n📦 DEPENDENCIES:")
    all_good &= check_import("gym", "OpenAI Gym")
    all_good &= check_import("numpy", "NumPy")
    all_good &= check_import("pybullet", "PyBullet Physics")
    all_good &= check_import("stable_baselines3", "Stable-Baselines3")
    
    try:
        from gym_pybullet_drones.envs.BaseAviary import BaseAviary
        print("✅ gym-pybullet-drones")
    except ImportError as e:
        print(f"❌ gym-pybullet-drones - IMPORT ERROR: {e}")
        all_good = False
    
    # Check environment functionality
    print("\n🧪 ENVIRONMENT TESTS:")
    try:
        from training_env_motor_control import MotorControlAviary
        env = MotorControlAviary()
        obs = env.reset()
        print(f"✅ Stage 2 Environment (Obs shape: {obs.shape})")
        env.close()
    except Exception as e:
        print(f"❌ Stage 2 Environment - ERROR: {e}")
        all_good = False
    
    try:
        from training_env_path_planning import PathPlanningAviary
        env = PathPlanningAviary()
        obs = env.reset()
        print(f"✅ Stage 1 Environment (Obs shape: {obs.shape})")
        env.close()
    except Exception as e:
        print(f"❌ Stage 1 Environment - ERROR: {e}")
        all_good = False
    
    # Check vision system
    print("\n👁️  VISION SYSTEM TESTS:")
    try:
        from stereo_vision import StereoVision
        from depth_downsampler import DepthDownsampler
        
        stereo = StereoVision(baseline=0.1)
        downsampler = DepthDownsampler(target_width=64, target_height=32)
        
        print(f"✅ Stereo Vision (Baseline: {stereo.baseline}m)")
        print(f"✅ Depth Downsampler (Output: {downsampler.target_width}x{downsampler.target_height})")
    except Exception as e:
        print(f"❌ Vision System - ERROR: {e}")
        all_good = False
    
    # Final summary
    print("\n" + "=" * 70)
    if all_good:
        print("🎉 ALL CHECKS PASSED - READY TO TRAIN!")
        print("=" * 70)
        print("\n📋 NEXT STEPS:")
        print("1. Train Stage 2 Motor Control:")
        print("   python train_motor_control.py --timesteps 500000 --name motor_control_v1")
        print("\n2. Monitor training:")
        print("   tensorboard --logdir ./logs/motor_control_v1")
        print("\n3. Review training strategy:")
        print("   cat TRAINING_STRATEGY.md")
        print("\n" + "=" * 70)
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
