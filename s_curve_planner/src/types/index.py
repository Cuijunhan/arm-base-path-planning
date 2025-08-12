# filepath: /s_curve_planner/s_curve_planner/src/types/index.py
from typing import Tuple, List

# Define types for joint trajectory data
JointTrajectory = Tuple[List[float], List[float], List[float], List[float]]
TrajectoryData = Tuple[List[float], List[float], List[float], List[float], List[float]]  # (positions, velocities, accelerations, jerks, time)