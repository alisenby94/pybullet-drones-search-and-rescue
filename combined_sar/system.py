"""
Consolidated Mission Planning + RL + LM System

ALL-IN-ONE unified system for drone SAR missions combining:
- Language model planning
- Reinforcement learning control
- Voxel mapping
- Person detection
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import requests

# Gym/PyBullet
try:
    import gymnasium as gym
    from gymnasium.spaces import Box
except ImportError:
    import gym
    from gym.spaces import Box

import pybullet as p
import pybullet_data

# OpenCV for video recording
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[Warning] cv2 not available - video recording disabled")

# PyBullet Drones core
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

# Local imports - use relative imports within package
from .voxel_mapper import PerpendicularBand2DMapper
from .waypoint_planner import WaypointPlanner
from .person_detector import PersonDetector
from .lm_client import LMClient


class MappingEnvironment(VelocityAviary):
    """Consolidated mapping environment with velocity control and voxel mapping"""
    
    def __init__(
        self,
        drone_model=DroneModel.CF2X,
        num_drones=1,
        gui=False,
        record=False,
        ctrl_freq=48,
        pyb_freq=240,
    ):
        # Initialize voxel mapper
        hangar_centers = [(10.0, 10.0), (-10.0, 10.0), (10.0, -10.0), (-10.0, -10.0)]
        half_size = 4.0
        margin = 5.0
        allowed_regions = []
        for (cx, cy) in hangar_centers:
            min_xy = (cx - (half_size + margin), cy - (half_size + margin))
            max_xy = (cx + (half_size + margin), cy + (half_size + margin))
            allowed_regions.append((min_xy, max_xy))
        
        self.voxel_mapper = PerpendicularBand2DMapper(
            world_size=60.0,
            resolution=0.25,
            pixels_half_width=2,
            debug=False,
            height_band=5.0,
            allowed_regions=allowed_regions,
        )
        
        # Camera parameters
        self.CAM_FOV = 60.0
        self.CAM_WIDTH = 64
        self.CAM_HEIGHT = 48
        
        # Initialize parent environment with drone starting 1m in air
        initial_pos = np.array([[0.0, 0.0, 1.0]])
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=np.inf,
            initial_xyzs=initial_pos,
            initial_rpys=None,
            physics=Physics.PYB_GND_DRAG_DW,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=True,
            user_debug_gui=False,
        )
        
        # Set speed limit for RL compatibility
        self.SPEED_LIMIT = 2.0
        
        # Vision buffers
        self.IMG_RES = np.array([self.CAM_WIDTH, self.CAM_HEIGHT])
        self.rgb = np.zeros((self.NUM_DRONES, self.CAM_HEIGHT, self.CAM_WIDTH, 4))
        self.dep = np.ones((self.NUM_DRONES, self.CAM_HEIGHT, self.CAM_WIDTH))
    
    def _addObstacles(self):
        """Add hangars and person"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        
        # Get asset directory - assets folder is inside combined_sar package
        asset_dir = Path(__file__).parent / "assets"
        print(f"[Environment] Asset directory: {asset_dir}")
        
        # Building positions for exclusion zones
        self.hangar_positions = [(10, 10), (-10, 10), (10, -10), (-10, -10)]
        hangar_half_size = 4.5  # Hangars are 8.5x8.5m
        
        # Load hangar URDFs
        hangar_urdf = asset_dir / "hangar" / "hangar6.urdf"
        if hangar_urdf.exists():
            print(f"[Environment] Loading hangars from {hangar_urdf}")
            for pos in self.hangar_positions:
                try:
                    p.loadURDF(str(hangar_urdf), basePosition=[pos[0], pos[1], 0.0], useFixedBase=True, physicsClientId=self.CLIENT)
                    print(f"[Environment] Hangar placed at ({pos[0]}, {pos[1]})")
                except Exception as e:
                    print(f"[Environment] Failed to load hangar at {pos}: {e}")
        else:
            print(f"[Environment] WARNING: Hangar URDF not found at {hangar_urdf}")
        
        # Add person at random location, avoiding buildings
        try:
            half_world = self.voxel_mapper.world_size / 2.0 - 1.0
            rng = np.random.default_rng()
            
            # Keep trying until we find a valid position
            max_attempts = 100
            for attempt in range(max_attempts):
                px = float(rng.uniform(-half_world, half_world))
                py = float(rng.uniform(-half_world, half_world))
                
                # Check if inside any hangar
                inside_hangar = False
                for hx, hy in self.hangar_positions:
                    if abs(px - hx) < hangar_half_size and abs(py - hy) < hangar_half_size:
                        inside_hangar = True
                        break
                
                if not inside_hangar:
                    break
            
            self.person_pos = (px, py)  # Store for later reference
            
            # Load person URDF
            person_urdf = asset_dir / "people" / "person1.urdf"
            if person_urdf.exists():
                print(f"[Environment] Loading person from {person_urdf}")
                p.loadURDF(str(person_urdf), basePosition=[px, py, 0.0], useFixedBase=True, physicsClientId=self.CLIENT)
                print(f"[Environment] Person placed at ({px:.2f}, {py:.2f})")
            else:
                print(f"[Environment] WARNING: Person URDF not found at {person_urdf}")
                
        except Exception as e:
            print(f"[Environment] Error placing person: {e}")
            import traceback
            traceback.print_exc()
    
    def _getDroneImages(self, drone_id=0, segmentation=False):
        """Get drone camera images"""
        try:
            # Get camera matrix and depth
            if hasattr(super(), '_getDroneImages'):
                rgb, depth, seg = super()._getDroneImages(drone_id, segmentation)
                # Convert RGBA to RGB if needed
                if rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]
                return rgb, depth, seg
        except Exception as e:
            pass
        
        # Fallback: return dummy data with RGB (3 channels)
        depth = np.ones((self.CAM_HEIGHT, self.CAM_WIDTH)) * 5.0
        rgb = np.zeros((self.CAM_HEIGHT, self.CAM_WIDTH, 3), dtype=np.uint8)
        seg = np.zeros((self.CAM_HEIGHT, self.CAM_WIDTH), dtype=np.int32)
        return rgb, depth, seg
    
    def update_voxel_map(self):
        """Update voxel map with depth"""
        try:
            depth = self._getDroneImages(0)[1]
            self.voxel_mapper.integrate_depth(depth, self.pos[0], self.rpy[0])
        except:
            pass


class UnifiedMissionSystem:
    """Complete mission system with LM planning, RL control, and mapping"""
    
    def __init__(
        self,
        duration=60,
        control_mode="hybrid",
        lm_server_url="http://localhost:8000",
        gui=False,
        record=False,
        yolo_confidence=0.5,
        terminate_on_person=False,
        output_dir="results",
    ):
        self.duration = duration
        self.control_mode = control_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        print("[System] Initializing environment...")
        self.env = MappingEnvironment(gui=gui, record=record, ctrl_freq=48, pyb_freq=240)
        
        # Initialize components
        print("[System] Initializing components...")
        self.waypoint_planner = WaypointPlanner(hover_alt=1.0, max_step=2.0)  # 2.0m step = fewer waypoints
        self.person_detector = PersonDetector(confidence=yolo_confidence)
        self.lm_client = LMClient(server_url=lm_server_url)
        
        # PID controller
        self.pid_controller = DSLPIDControl(drone_model=DroneModel.CF2X)
        
        # Mission state
        self.step_count = 0
        self.current_waypoints: List[np.ndarray] = []
        self.waypoint_idx = 0
        self.action_history: List[Dict] = []
        self.last_replan_time = 0.0
        self.replan_period = 3.0  # Replan every 3 seconds (less frequent)
        
        # Store GUI flag for reset
        self._gui = gui
        self._record = record
        self.terminate_on_person = terminate_on_person
        
        # Logging
        self.log_file = self.output_dir / "mission.log"
        self.lm_exchange_log = self.output_dir / "lm_exchange.log"
        self._log(f"[System] Initialized - duration={duration}s, mode={control_mode}")
        
        # State for step-based operation
        self.start_time = None
        self.episode_start_time = None
        self.mission_active = False
        self.person_detected = False
        self.drone_pos = np.array([0.0, 0.0, 1.0])
        self.lm_exchange_counter = 0
    
    def reset(self):
        """Reset mission state for new episode"""
        # Close current environment
        try:
            self.env.close()
        except:
            pass
        
        # Create fresh environment with stored GUI flag
        self.env = MappingEnvironment(gui=self._gui, record=self._record, ctrl_freq=48, pyb_freq=240)
        
        # Reset state
        self.step_count = 0
        self.current_waypoints = []
        self.waypoint_idx = 0
        self.action_history = []
        self.last_replan_time = 0.0
        self.person_detected = False
        self.episode_start_time = time.time()
        self.mission_active = True
        
        # Get initial observation
        obs, info = self.step(action=None)
        return obs, info
    
    @property
    def person_pos(self):
        """Get current person position"""
        try:
            return getattr(self.env, 'person_pos', None)
        except:
            return None
    
    def step(self, action=None):
        """Step the environment - single control loop iteration"""
        if not self.mission_active:
            return None, {"done": True, "person_detected": False}
        
        if self.episode_start_time is None:
            self.episode_start_time = time.time()
        
        t_elapsed = time.time() - self.episode_start_time
        
        # Get current drone state BEFORE computing action
        drone_pos = np.array(self.env.pos[0] if hasattr(self.env, 'pos') else [0, 0, 1], dtype=np.float32)
        drone_vel = np.array(self.env.vel[0] if hasattr(self.env, 'vel') else [0, 0, 0], dtype=np.float32)
        self.drone_pos = drone_pos
        
        # Request plan ONLY when all waypoints are exhausted (no time-based replanning)
        lm_response = None
        waypoints_exhausted = self.waypoint_idx >= len(self.current_waypoints)
        if waypoints_exhausted:
            try:
                obs_json = self._build_observation(drone_pos)
                history_json = json.dumps(self.action_history[-5:])
                plan = self.lm_client.get_plan(obs_json, history_json)
                
                # Log the exchange
                self._log_lm_exchange(
                    exchange_id=self.lm_exchange_counter,
                    prompt_observation=obs_json,
                    prompt_history=history_json,
                    response=plan,
                    status="success" if plan else "empty_response"
                )
                self.lm_exchange_counter += 1
                
                if plan and plan.get('actions'):
                    lm_response = plan
                    self._process_actions(plan.get("actions", []), drone_pos)
                else:
                    self._fallback_plan(drone_pos)
            except Exception as e:
                self._fallback_plan(drone_pos)
            
            self.last_replan_time = t_elapsed
        
        # Compute velocity command toward target waypoint
        if len(self.current_waypoints) > self.waypoint_idx:
            target = self.current_waypoints[self.waypoint_idx]
        else:
            target = np.array([drone_pos[0], drone_pos[1], 1.0])
        
        # Compute normalized velocity command (VelocityAviary format)
        # VelocityAviary expects: [direction_x, direction_y, direction_z, speed_fraction]
        # where first 3 are normalized direction, 4th is speed as fraction of max
        direction = target - drone_pos
        dist_3d = np.linalg.norm(direction)
        
        if dist_3d > 0.05:
            # Normalize direction to unit vector
            direction_normalized = direction / (dist_3d + 1e-6)
            # Speed fraction: full speed when far, slow down when close
            speed_fraction = min(1.0, dist_3d / 3.0)  # Full speed at 3m distance
            velocity_cmd = np.array([[
                direction_normalized[0],
                direction_normalized[1],
                direction_normalized[2],
                speed_fraction  # This is the speed, not yaw rate!
            ]])
        else:
            # At target, hover
            velocity_cmd = np.array([[0, 0, 0, 0]])
        
        # Execute environment step WITH the computed velocity command
        step_result = self.env.step(velocity_cmd)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done = step_result
            info = {}
        
        # Update drone position after step
        drone_pos = np.array(self.env.pos[0] if hasattr(self.env, 'pos') else [0, 0, 1], dtype=np.float32)
        self.drone_pos = drone_pos
        
        # Check waypoint reached (0.5m radius)
        if len(self.current_waypoints) > self.waypoint_idx:
            target = self.current_waypoints[self.waypoint_idx]
            dist = np.linalg.norm(target[:2] - drone_pos[:2])
            if dist < 0.5:
                self.waypoint_idx += 1
                self._log(f"[Waypoint] Reached {self.waypoint_idx}/{len(self.current_waypoints)}")
        
        # Update mapper
        try:
            self.env.update_voxel_map()
        except:
            pass
        
        # Check for person detection
        person_found = False
        try:
            rgb, _, _ = self.env._getDroneImages(0)
            detected, _ = self.person_detector.detect(rgb.astype(np.uint8))
            if detected:
                person_found = True
                self.person_detected = True
                self._log("[Step] Person detected!")
        except:
            pass
        
        self.step_count += 1
        
        # Return observation and info
        return obs, {
            "done": person_found or self.step_count > 10000,
            "person_detected": person_found,
            "drone_pos": drone_pos.tolist(),
            "lm_response": lm_response,
            "current_action": "waypoint_follow",
            "elapsed_time": t_elapsed,
            "velocity_cmd": velocity_cmd[0].tolist() if velocity_cmd is not None else [0,0,0,0]
        }
    
    def close(self):
        """Close environment"""
        try:
            self.env.close()
        except:
            pass
        self.mission_active = False
    
    def _log(self, msg):
        """Log message"""
        print(msg)
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {msg}\n")
        except:
            pass
    
    def _log_lm_exchange(self, exchange_id, prompt_observation, prompt_history, response, status):
        """Log LM exchange with full prompt and response"""
        try:
            with open(self.lm_exchange_log, 'a') as f:
                # Write exchange header
                f.write(f"\n{'='*80}\n")
                f.write(f"EXCHANGE #{exchange_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Status: {status}\n")
                f.write(f"{'='*80}\n\n")
                
                # Write prompt
                f.write("PROMPT - OBSERVATION:\n")
                f.write("-" * 80 + "\n")
                if isinstance(prompt_observation, str):
                    try:
                        obs_dict = json.loads(prompt_observation)
                        f.write(json.dumps(obs_dict, indent=2))
                    except:
                        f.write(prompt_observation)
                else:
                    f.write(str(prompt_observation))
                f.write("\n\n")
                
                f.write("PROMPT - HISTORY:\n")
                f.write("-" * 80 + "\n")
                if isinstance(prompt_history, str):
                    try:
                        history_list = json.loads(prompt_history)
                        f.write(json.dumps(history_list, indent=2))
                    except:
                        f.write(prompt_history)
                else:
                    f.write(str(prompt_history))
                f.write("\n\n")
                
                # Write response
                f.write("RESPONSE:\n")
                f.write("-" * 80 + "\n")
                if response:
                    if isinstance(response, dict):
                        f.write(json.dumps(response, indent=2))
                    else:
                        f.write(str(response))
                else:
                    f.write("(No response)")
                f.write("\n\n")
        except Exception as e:
            print(f"[Error] Failed to log LM exchange: {e}")
    
    def run(self):
        """Main mission loop with video recording"""
        start_time = time.time()
        ctrl_freq = 48  # Control frequency
        ctrl_dt = 1.0 / ctrl_freq
        next_render = 0
        
        self._log(f"\n[Mission] Starting {self.duration}s mission")
        
        # Video recording setup
        video_writer = None
        video_path = self.output_dir / "mission_video.mp4"
        video_fps = 24  # Lower FPS for less overhead
        video_width, video_height = 640, 480  # Smaller resolution for less overhead
        frame_interval = max(1, int(ctrl_freq / video_fps))
        
        if self._record and CV2_AVAILABLE:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(video_path), fourcc, float(video_fps), (video_width, video_height))
                if video_writer.isOpened():
                    self._log(f"[Video] Recording to {video_path}")
                else:
                    video_writer = None
                    self._log("[Video] Failed to open video writer")
            except Exception as e:
                self._log(f"[Video] Setup failed: {e}")
        
        # Camera parameters for third-person view
        cam_dist = 12.0  # Farther out for better overview
        cam_yaw = 45.0
        cam_pitch = -35.0
        
        # Timing for real-time sync
        sim_start_time = time.time()
        
        try:
            while (time.time() - start_time) < self.duration:
                t_elapsed = time.time() - start_time
                step_start = time.time()
                
                # Get current position BEFORE computing action
                drone_pos = np.array(self.env.pos[0] if hasattr(self.env, 'pos') else [0, 0, 1], dtype=np.float32)
                drone_vel = np.array(self.env.vel[0] if hasattr(self.env, 'vel') else [0, 0, 0], dtype=np.float32)
                drone_rpy = np.array(self.env.rpy[0] if hasattr(self.env, 'rpy') else [0, 0, 0], dtype=np.float32)
                
                # Request new plan ONLY when all waypoints are exhausted
                waypoints_exhausted = self.waypoint_idx >= len(self.current_waypoints)
                if waypoints_exhausted:
                    self._request_plan(drone_pos)
                
                # Get target waypoint
                if len(self.current_waypoints) > self.waypoint_idx:
                    target = self.current_waypoints[self.waypoint_idx]
                else:
                    target = np.array([drone_pos[0], drone_pos[1], 1.0])
                
                # Compute normalized velocity command toward target
                # VelocityAviary expects: [direction_x, direction_y, direction_z, speed_fraction]
                direction = target - drone_pos
                dist_3d = np.linalg.norm(direction)
                
                if dist_3d > 0.05:
                    # Normalize direction to unit vector
                    direction_normalized = direction / (dist_3d + 1e-6)
                    # Speed fraction: full speed when far, slow down when close
                    speed_fraction = min(1.0, dist_3d / 3.0)  # Full speed at 3m distance
                    velocity_cmd = np.array([[
                        direction_normalized[0],
                        direction_normalized[1],
                        direction_normalized[2],
                        speed_fraction  # This is the speed, not yaw rate!
                    ]])
                else:
                    velocity_cmd = np.array([[0, 0, 0, 0]])
                
                # Step environment WITH the computed velocity command
                step_result = self.env.step(velocity_cmd)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                else:
                    obs, reward, done = step_result
                
                # Update position after step
                drone_pos = np.array(self.env.pos[0] if hasattr(self.env, 'pos') else [0, 0, 1], dtype=np.float32)
                
                # Check waypoint reached (0.5m radius)
                if len(self.current_waypoints) > self.waypoint_idx:
                    target = self.current_waypoints[self.waypoint_idx]
                    dist = np.linalg.norm(target[:2] - drone_pos[:2])
                    if dist < 0.5:
                        self.waypoint_idx += 1
                        if self.waypoint_idx < len(self.current_waypoints):
                            self._log(f"[Waypoint] Reached {self.waypoint_idx}/{len(self.current_waypoints)}")
                
                # Update mapper (throttled - every 10 steps to reduce overhead)
                if self.step_count % 10 == 0:
                    try:
                        self.env.update_voxel_map()
                    except:
                        pass
                
                # Check for person (throttled - every 24 steps = ~0.5 seconds)
                if self.terminate_on_person and self.step_count % 24 == 0:
                    try:
                        rgb, _, _ = self.env._getDroneImages(0)
                        detected, _ = self.person_detector.detect(rgb.astype(np.uint8))
                        if detected:
                            self._log("[Mission] Person detected! Terminating.")
                            break
                    except:
                        pass
                
                # Render and record video
                if t_elapsed >= next_render:
                    # Update GUI camera to follow drone
                    if self._gui:
                        try:
                            cam_target = drone_pos.tolist()
                            p.resetDebugVisualizerCamera(
                                cameraDistance=cam_dist,
                                cameraYaw=cam_yaw,
                                cameraPitch=cam_pitch,
                                cameraTargetPosition=cam_target,
                                physicsClientId=self.env.CLIENT
                            )
                        except:
                            pass
                    
                    self.env.render()
                    next_render = t_elapsed + 1.0 / 30
                
                # Record video frame (throttled)
                if video_writer is not None and (self.step_count % frame_interval == 0):
                    try:
                        cam_target = drone_pos.tolist()
                        view = p.computeViewMatrixFromYawPitchRoll(
                            cameraTargetPosition=cam_target,
                            distance=cam_dist,
                            yaw=cam_yaw,
                            pitch=cam_pitch,
                            roll=0,
                            upAxisIndex=2
                        )
                        proj = p.computeProjectionMatrixFOV(
                            fov=60.0,
                            aspect=video_width/video_height,
                            nearVal=0.1,
                            farVal=100.0
                        )
                        _, _, rgba, _, _ = p.getCameraImage(
                            width=video_width,
                            height=video_height,
                            viewMatrix=view,
                            projectionMatrix=proj,
                            physicsClientId=self.env.CLIENT
                        )
                        rgb = np.reshape(rgba, (video_height, video_width, 4))[:, :, :3].astype(np.uint8)
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        video_writer.write(bgr)
                    except Exception as e:
                        if self.step_count % 240 == 0:
                            self._log(f"[Video] Frame capture error: {e}")
                
                self.step_count += 1
                
                # Real-time sync - wait if running faster than real-time
                step_duration = time.time() - step_start
                if step_duration < ctrl_dt:
                    time.sleep(ctrl_dt - step_duration)
            
            self._log(f"[Mission] Completed {self.step_count} steps in {t_elapsed:.1f}s")
            self._save_results()
        
        finally:
            # Release video writer
            if video_writer is not None:
                video_writer.release()
                self._log(f"[Video] Saved to {video_path}")
            self._cleanup()
    
    def _request_plan(self, drone_pos):
        """Request plan from LM server, with fallback to local planning"""
        try:
            obs_json = self._build_observation(drone_pos)
            history_json = json.dumps(self.action_history[-5:])
            
            # Get plan from LM server
            plan = self.lm_client.get_plan(obs_json, history_json)
            
            # Log the exchange
            self._log_lm_exchange(
                exchange_id=self.lm_exchange_counter,
                prompt_observation=obs_json,
                prompt_history=history_json,
                response=plan,
                status="success" if plan else "empty_response"
            )
            self.lm_exchange_counter += 1
            
            if plan and plan.get('actions'):
                self._log(f"[LM] Received plan with {len(plan.get('actions', []))} actions")
                self._process_actions(plan.get("actions", []), drone_pos)
            else:
                # LM server not available, use fallback exploration
                self._fallback_plan(drone_pos)
        except Exception as e:
            # Log the failed exchange
            self._log_lm_exchange(
                exchange_id=self.lm_exchange_counter,
                prompt_observation=obs_json if 'obs_json' in locals() else "unknown",
                prompt_history=history_json if 'history_json' in locals() else "unknown",
                response=None,
                status=f"error: {str(e)}"
            )
            self.lm_exchange_counter += 1
            
            # LM server error, use fallback exploration
            self._fallback_plan(drone_pos)
    
    def _fallback_plan(self, drone_pos):
        """Generate fallback exploration waypoints when LM not available"""
        import random
        
        # Generate exploration pattern
        direction = random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        distance = random.uniform(5, 15)
        
        waypoints = self.waypoint_planner.plan_explore(drone_pos, direction, distance)
        
        self.current_waypoints = waypoints
        self.waypoint_idx = 0
        
        self._log(f"[Fallback] Generated exploration waypoints: {len(waypoints)} points")
    
    def _process_actions(self, actions, drone_pos):
        """Convert actions to waypoints"""
        self.current_waypoints = []
        self.waypoint_idx = 0
        
        for action in actions:
            a_type = action.get("action", "")
            params = action.get("parameters", {})
            
            if a_type == "scan_surroundings":
                wps = self.waypoint_planner.plan_scan(drone_pos)
                self.current_waypoints.extend(wps)
            elif a_type == "explore":
                direction = params.get("direction", "N")
                distance = min(params.get("distance", 10), 3.0)
                wps = self.waypoint_planner.plan_explore(drone_pos, direction, distance)
                self.current_waypoints.extend(wps)
            elif a_type == "explore_obstacle":
                direction = params.get("direction", "N")
                wps = self.waypoint_planner.plan_explore_obstacle(drone_pos, direction)
                self.current_waypoints.extend(wps)
        
        self._log(f"[Plan] Generated {len(self.current_waypoints)} waypoints")
    
    def _build_observation(self, drone_pos):
        """Build observation JSON"""
        grid = self.env.voxel_mapper.grid if hasattr(self.env, 'voxel_mapper') else np.zeros((200, 200))
        obs = {
            "drone_position": {"x": float(drone_pos[0]), "y": float(drone_pos[1]), "z": float(drone_pos[2])},
            "obstacles": [],
            "occupied_cells": int(np.count_nonzero(grid)) if grid.size > 0 else 0,
        }
        return json.dumps(obs)
    
    def _save_results(self):
        """Save results"""
        try:
            results_dir = self.output_dir / "latest"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            if hasattr(self.env, 'voxel_mapper'):
                self.env.voxel_mapper.export_png(str(results_dir / "voxel_map.png"))
            
            summary = {
                "duration": self.duration,
                "steps": self.step_count,
                "waypoints": len(self.current_waypoints),
            }
            with open(results_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self._log(f"[System] Save error: {e}")
    
    def _cleanup(self):
        """Cleanup"""
        try:
            self.env.close()
        except:
            pass
        self._log("[System] Cleanup complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--control-mode", default="hybrid")
    parser.add_argument("--lm-server-url", default="http://localhost:8000")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--yolo-confidence", type=float, default=0.5)
    parser.add_argument("--terminate-on-person", action="store_true")
    parser.add_argument("--output-dir", default="results")
    
    args = parser.parse_args()
    
    system = UnifiedMissionSystem(**vars(args))
    system.run()


if __name__ == "__main__":
    main()
