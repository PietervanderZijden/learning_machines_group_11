import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "catkin_ws/src/learning_machines/src"))
sys.path.insert(0, str(ROOT / "catkin_ws/src/robobo_interface/src"))
