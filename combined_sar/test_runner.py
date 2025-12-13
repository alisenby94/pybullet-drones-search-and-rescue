#!/usr/bin/env python3
"""
Multiple runs test for drone search and rescue with video recording and metrics.

Usage:
    python -m combined_sar.test_runner --runs 5 --timeout 120
    python -m combined_sar.test_runner --runs 3 --timeout 60 --no-video
"""

import os
import json
import time
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2
from collections import defaultdict
import pybullet as p

from .system import UnifiedMissionSystem
from .utils import get_logger


class TestRunner:
    def __init__(self, num_runs=5, timeout_seconds=120, record_video=True, lm_server_url="http://localhost:8000"):
        self.num_runs = num_runs
        self.timeout = timeout_seconds
        self.record_video = record_video
        self.lm_server_url = lm_server_url
        self.logger = get_logger("TestRunner")
        
        # Create results directory
        self.results_dir = Path(__file__).parent / "results" / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "total_runs": num_runs,
            "successful_runs": 0,
            "failed_runs": 0,
            "timeout_runs": 0,
            "runs": [],
            "summary": {}
        }
        
        self.lm_exchanges_all = []
        
    def run_single_mission(self, run_id, env, frame_data=None):
        """Run single mission and track metrics"""
        start_time = time.time()
        run_metrics = {
            "run_id": run_id,
            "start_time": start_time,
            "status": "running",
            "detection_time": None,
            "total_steps": 0,
            "actions_count": 0,
            "lm_exchanges": [],
            "person_position": None,
            "final_position": None
        }
        
        try:
            # Reset environment for new mission
            obs, info = env.reset()
            person_pos = getattr(env, 'person_pos', None)
            run_metrics["person_position"] = person_pos
            
            detected = False
            
            # Video recording setup - camera follows drone
            video_writer = None
            video_path = self.results_dir / f"run_{run_id:03d}_video.mp4"
            video_width, video_height = 1280, 720
            frame_interval = 2  # Capture every 2 steps for 24fps output
            # Camera settings: distance, yaw (rotation around z), pitch (angle down)
            cam_dist = 12.0   # Distance from drone (meters)
            cam_yaw = 45.0    # Rotate 45 degrees for nice angle
            cam_pitch = -45.0 # Look down at 45 degrees for good overhead view
            
            if self.record_video:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 24.0, (video_width, video_height))
                    if video_writer.isOpened():
                        self.logger.info(f"[Run {run_id}] Recording video to {video_path}")
                    else:
                        video_writer = None
                except Exception as e:
                    self.logger.warning(f"[Run {run_id}] Video setup failed: {e}")
            
            # Mission loop
            while time.time() - start_time < self.timeout:
                # Execute step
                obs, info = env.step(action=None)
                
                # Check if person detected
                if info.get('person_detected', False):
                    detection_time = time.time() - start_time
                    run_metrics["detection_time"] = detection_time
                    run_metrics["status"] = "success"
                    detected = True
                    self.logger.info(f"[Run {run_id}] Person detected in {detection_time:.2f}s")
                    break
                
                # Get LM plan if available
                if info.get('lm_response'):
                    run_metrics["lm_exchanges"].append({
                        "timestamp": time.time() - start_time,
                        "response": info.get('lm_response'),
                        "action": info.get('current_action')
                    })
                    run_metrics["actions_count"] += 1
                
                run_metrics["total_steps"] += 1
                
                # Capture video frame using PyBullet camera centered on drone
                if video_writer is not None and (run_metrics["total_steps"] % frame_interval == 0):
                    try:
                        # Get drone position from step info (most accurate)
                        drone_pos = info.get('drone_pos', None)
                        if drone_pos is None:
                            drone_pos = getattr(env, 'drone_pos', [0, 0, 1])
                        if hasattr(drone_pos, 'tolist'):
                            drone_pos = drone_pos.tolist()
                        
                        # Camera follows drone from above and behind
                        cam_target = [float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2])]
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
                            physicsClientId=env.env.CLIENT
                        )
                        rgb = np.reshape(rgba, (video_height, video_width, 4))[:, :, :3].astype(np.uint8)
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        video_writer.write(bgr)
                    except Exception as e:
                        if run_metrics["total_steps"] % 100 == 0:
                            self.logger.debug(f"[Run {run_id}] Frame capture error: {e}")
            
            # Check timeout
            if not detected:
                run_metrics["status"] = "timeout"
                self.logger.warning(f"[Run {run_id}] Timeout - person not detected in {self.timeout}s")
            
            # Get final position
            drone_pos = getattr(env, 'drone_pos', None)
            run_metrics["final_position"] = drone_pos.tolist() if hasattr(drone_pos, 'tolist') else drone_pos
            run_metrics["end_time"] = time.time()
            run_metrics["elapsed_time"] = run_metrics["end_time"] - start_time
            
            # Release video writer
            if video_writer is not None:
                video_writer.release()
                self.logger.info(f"[Run {run_id}] Video saved: {video_path}")
            
        except Exception as e:
            run_metrics["status"] = "error"
            run_metrics["error"] = str(e)
            self.logger.error(f"[Run {run_id}] Error: {e}")
        
        return run_metrics
    
    def _copy_lm_exchange_log(self, run_output_dir, run_id):
        """Copy LM exchange log from run directory to results root"""
        try:
            src_log = Path(run_output_dir) / "lm_exchange.log"
            dst_log = self.results_dir / f"run_{run_id:03d}_lm_exchange.log"
            
            if src_log.exists():
                shutil.copy2(str(src_log), str(dst_log))
                self.logger.info(f"[Run {run_id}] LM exchange log copied: {dst_log}")
            else:
                self.logger.debug(f"[Run {run_id}] No LM exchange log found at {src_log}")
        except Exception as e:
            self.logger.warning(f"[Run {run_id}] Failed to copy LM exchange log: {e}")
    
    def _save_voxel_map(self, env, run_id, run_metrics):
        """Save voxel map visualization with annotations"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            # Get the voxel grid
            if not hasattr(env, 'env') or not hasattr(env.env, 'voxel_mapper'):
                self.logger.debug(f"[Run {run_id}] No voxel mapper found")
                return
            
            voxel_mapper = env.env.voxel_mapper
            grid = voxel_mapper.grid
            world_size = voxel_mapper.world_size
            
            # Create figure with enhanced visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            extent = (-world_size/2, world_size/2, -world_size/2, world_size/2)
            
            # Plot occupancy grid (flip vertically for correct orientation)
            ax.imshow(grid.T, origin='lower', cmap='gray_r', extent=extent, alpha=0.7)
            
            # Plot hangar positions (4 corners)
            hangar_positions = [(10, 10), (-10, 10), (10, -10), (-10, -10)]
            hangar_half_size = 4.25
            for (hx, hy) in hangar_positions:
                rect = Rectangle(
                    (hx - hangar_half_size, hy - hangar_half_size),
                    hangar_half_size * 2, hangar_half_size * 2,
                    fill=False, edgecolor='blue', linewidth=2, linestyle='--',
                    label='Hangar' if (hx, hy) == hangar_positions[0] else ''
                )
                ax.add_patch(rect)
            
            # Plot person position if available
            person_pos = run_metrics.get("person_position")
            if person_pos is not None:
                if hasattr(person_pos, '__len__') and len(person_pos) >= 2:
                    ax.scatter(person_pos[0], person_pos[1], c='red', s=200, marker='*', 
                              zorder=10, label='Person', edgecolors='white', linewidth=1)
            
            # Plot drone start and end positions
            ax.scatter(0, 0, c='green', s=150, marker='^', zorder=10, 
                      label='Start', edgecolors='white', linewidth=1)
            
            final_pos = run_metrics.get("final_position")
            if final_pos is not None:
                if hasattr(final_pos, '__len__') and len(final_pos) >= 2:
                    ax.scatter(final_pos[0], final_pos[1], c='orange', s=150, marker='s',
                              zorder=10, label='End', edgecolors='white', linewidth=1)
            
            # Add title and labels
            status = run_metrics.get("status", "unknown")
            detection_time = run_metrics.get("detection_time", None)
            title = f"Run {run_id}: {status.upper()}"
            if detection_time:
                title += f" ({detection_time:.1f}s)"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Add colorbar
            occupied_cells = np.count_nonzero(grid)
            ax.text(0.02, 0.98, f"Occupied cells: {occupied_cells}", transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save figure
            voxel_map_path = self.results_dir / f"run_{run_id:03d}_voxel_map.png"
            plt.savefig(voxel_map_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"[Run {run_id}] Voxel map saved: {voxel_map_path}")
            
        except Exception as e:
            self.logger.warning(f"[Run {run_id}] Failed to save voxel map: {e}")
    
    def run_tests(self):
        """Execute multiple test runs"""
        self.logger.info(f"Starting {self.num_runs} test runs")
        self.logger.info(f"Results directory: {self.results_dir}")
        
        for run_id in range(self.num_runs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Starting Run {run_id + 1}/{self.num_runs}")
            self.logger.info(f"{'='*60}")
            
            run_output_dir = str(self.results_dir / f"run_{run_id:03d}")
            
            try:
                env = UnifiedMissionSystem(
                    duration=self.timeout,
                    control_mode="hybrid",
                    lm_server_url=self.lm_server_url,
                    gui=False,
                    record=self.record_video,  # Enable recording if requested
                    yolo_confidence=0.5,
                    terminate_on_person=True,
                    output_dir=run_output_dir
                )
                
                run_metrics = self.run_single_mission(run_id, env)
                self.metrics["runs"].append(run_metrics)
                
                # Track status
                if run_metrics["status"] == "success":
                    self.metrics["successful_runs"] += 1
                elif run_metrics["status"] == "timeout":
                    self.metrics["timeout_runs"] += 1
                else:
                    self.metrics["failed_runs"] += 1
                
                # Store LM exchanges
                for exchange in run_metrics.get("lm_exchanges", []):
                    self.lm_exchanges_all.append({
                        "run_id": run_id,
                        **exchange
                    })
                
                # Copy LM exchange log file to results directory
                self._copy_lm_exchange_log(run_output_dir, run_id)
                
                # Save voxel map visualization
                self._save_voxel_map(env, run_id, run_metrics)
                
                # Close environment
                env.close()
                
                time.sleep(0.5)  # Small delay between runs
                
            except Exception as e:
                self.logger.error(f"Run {run_id} failed: {e}")
                self.metrics["failed_runs"] += 1
        
        # Generate report
        self._generate_report()
    
    def _generate_report(self):
        """Generate comprehensive test report"""
        success_rate = self.metrics["successful_runs"] / self.metrics["total_runs"] * 100
        
        # Calculate statistics
        successful_times = [r["detection_time"] for r in self.metrics["runs"] 
                          if r["detection_time"] is not None]
        avg_detection_time = np.mean(successful_times) if successful_times else 0
        min_detection_time = min(successful_times) if successful_times else 0
        max_detection_time = max(successful_times) if successful_times else 0
        
        total_actions = sum(r.get("actions_count", 0) for r in self.metrics["runs"])
        total_steps = sum(r.get("total_steps", 0) for r in self.metrics["runs"])
        
        self.metrics["summary"] = {
            "success_rate_percent": round(success_rate, 2),
            "successful_runs": self.metrics["successful_runs"],
            "failed_runs": self.metrics["failed_runs"],
            "timeout_runs": self.metrics["timeout_runs"],
            "avg_detection_time_seconds": round(avg_detection_time, 2),
            "min_detection_time_seconds": round(min_detection_time, 2),
            "max_detection_time_seconds": round(max_detection_time, 2),
            "total_actions": total_actions,
            "total_steps": total_steps,
            "avg_actions_per_run": round(total_actions / self.metrics["total_runs"], 2),
            "avg_steps_per_run": round(total_steps / self.metrics["total_runs"], 2),
            "num_lm_exchanges": len(self.lm_exchanges_all)
        }
        
        # Save metrics JSON
        metrics_path = self.results_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        self.logger.info(f"Metrics saved: {metrics_path}")
        
        # Save LM exchanges
        lm_path = self.results_dir / "lm_exchanges.json"
        with open(lm_path, 'w') as f:
            json.dump(self.lm_exchanges_all, f, indent=2, default=str)
        self.logger.info(f"LM exchanges saved: {lm_path}")
        
        # Generate text report
        report_path = self.results_dir / "test_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DRONE SEARCH & RESCUE TEST REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().isoformat()}\n")
            f.write(f"Results Directory: {self.results_dir}\n")
            f.write(f"Test Configuration:\n")
            f.write(f"  - Total Runs: {self.metrics['total_runs']}\n")
            f.write(f"  - Timeout per Run: {self.timeout} seconds\n")
            f.write(f"  - Video Recording: {'Yes' if self.record_video else 'No'}\n\n")
            
            f.write("SUMMARY METRICS\n")
            f.write("-"*70 + "\n")
            for key, value in self.metrics["summary"].items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n\nPER-RUN DETAILS\n")
            f.write("-"*70 + "\n")
            for run in self.metrics["runs"]:
                f.write(f"\nRun {run['run_id']}:\n")
                f.write(f"  Status: {run['status']}\n")
                if run.get("person_position"):
                    f.write(f"  Person Position: {run['person_position']}\n")
                if run.get("detection_time"):
                    f.write(f"  Detection Time: {run['detection_time']:.2f}s\n")
                f.write(f"  Total Steps: {run['total_steps']}\n")
                f.write(f"  LM Actions: {run['actions_count']}\n")
                if run.get("error"):
                    f.write(f"  Error: {run['error']}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        self.logger.info(f"Report saved: {report_path}")
        
        # Print summary to console
        print("\n" + "="*70)
        print("TEST EXECUTION COMPLETE")
        print("="*70)
        print(f"\nSuccess Rate: {success_rate:.1f}% ({self.metrics['successful_runs']}/{self.metrics['total_runs']})")
        print(f"Average Detection Time: {avg_detection_time:.2f}s")
        print(f"Total LM Exchanges: {len(self.lm_exchanges_all)}")
        print(f"\nResults saved to: {self.results_dir}")
        print("="*70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run multiple SAR test missions")
    parser.add_argument("--runs", type=int, default=5, help="Number of test runs (default: 5)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per run in seconds (default: 120)")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--lm-server-url", default="http://localhost:8000", help="LM server URL")
    
    args = parser.parse_args()
    
    runner = TestRunner(
        num_runs=args.runs,
        timeout_seconds=args.timeout,
        record_video=not args.no_video,
        lm_server_url=args.lm_server_url
    )
    
    runner.run_tests()


if __name__ == "__main__":
    main()
