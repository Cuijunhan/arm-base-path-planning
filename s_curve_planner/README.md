# S-Curve Planner and Visualizer

This project provides functionality for generating joint trajectories using S-curve interpolation and visualizing those trajectories. It is designed to facilitate motion planning for robotic joints, ensuring smooth transitions between specified positions.

## Project Structure

```
s_curve_planner
├── src
│   ├── planner.py          # Contains the SCurvePlanner class for trajectory generation
│   ├── visualizer.py       # Contains the TrajectoryVisualizer class for plotting trajectories
│   └── types
│       └── index.py       # Defines necessary types and constants
├── tests
│   └── test_planner.py     # Unit tests for the SCurvePlanner class
├── requirements.txt         # Lists project dependencies
├── setup.py                 # Package configuration and metadata
└── README.md                # Project documentation
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

### SCurvePlanner

The `SCurvePlanner` class is responsible for generating joint trajectories. You can create an instance of this class and use its methods to build normalized S-curves, calculate minimum times for joint movements, and generate trajectories.

Example:

```python
from src.planner import SCurvePlanner

planner = SCurvePlanner()
# Use planner methods to generate trajectories
```

### TrajectoryVisualizer

The `TrajectoryVisualizer` class allows you to visualize the generated trajectories. You can plot joint positions, velocities, accelerations, and jerks.

Example:

```python
from src.visualizer import TrajectoryVisualizer

visualizer = TrajectoryVisualizer()
# Use visualizer methods to plot trajectories
```

## Testing

Unit tests for the `SCurvePlanner` class are located in the `tests/test_planner.py` file. You can run the tests using:

```
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.