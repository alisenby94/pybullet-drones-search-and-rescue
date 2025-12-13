# Combined SAR System

Unified drone search and rescue system combining Language Model planning, Reinforcement Learning control, Voxel mapping, and Person detection.

## Quick Start

### Single Mission with GUI

```bash
cd /path/to/pybullet-drones-search-and-rescue
python -m combined_sar.system --gui --duration 60 --lm-server-url http://localhost:8000
```

### Multiple Test Runs

```bash
# Run 5 tests with 120s timeout each, with video recording
python -m combined_sar.test_runner --runs 5 --timeout 120

# Run 3 tests without video
python -m combined_sar.test_runner --runs 3 --timeout 60 --no-video

# With custom LM server
python -m combined_sar.test_runner --runs 5 --lm-server-url http://your-server:8000
```

## Components

- **system.py** - Main unified mission system

  - `MappingEnvironment` - PyBullet environment with voxel mapping
  - `UnifiedMissionSystem` - Complete mission controller

- **voxel_mapper.py** - 2D occupancy grid mapping from depth
- **waypoint_planner.py** - Convert LM actions to waypoint sequences
- **person_detector.py** - YOLO-based person detection
- **lm_client.py** - LM planning server client
- **test_runner.py** - Batch testing with metrics and video recording

## Output

Results are saved to `combined_sar/results/test_YYYYMMDD_HHMMSS/`:

- `test_report.txt` - Human-readable summary
- `test_metrics.json` - Detailed per-run metrics
- `run_XXX_video.mp4` - Video recording for each run
- `run_XXX_voxel_map.png` - Voxel map visualization
- `run_XXX_lm_exchange.log` - LM conversation logs

## Assets

The `assets/` folder contains:

- `hangar/hangar6.urdf` - Building URDF
- `people/person1.urdf` - Person URDF

## Dependencies

- gym-pybullet-drones
- ultralytics (for YOLO)
- opencv-python
- matplotlib
- requests
- numpy
