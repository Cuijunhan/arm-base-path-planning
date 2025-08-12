import matplotlib.pyplot as plt
import numpy as np

class SCurveVisualizer:
    @staticmethod
    def plot_trajectory(t_array, q_all, dq_all, ddq_all, dddq_all, joint_names=None):
        n_joints = q_all.shape[0]
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        axs[0].set_title("Joint Position (deg)")
        for i in range(n_joints):
            label = joint_names[i] if joint_names else f"J{i+1}"
            axs[0].plot(t_array, np.rad2deg(q_all[i]), label=label)
        axs[0].legend(loc='upper right', ncol=2)
        axs[0].grid(True)
        axs[1].set_title("Joint Velocity (rad/s)")
        for i in range(n_joints):
            axs[1].plot(t_array, dq_all[i])
        axs[1].grid(True)
        axs[2].set_title("Joint Acceleration (rad/s²)")
        for i in range(n_joints):
            axs[2].plot(t_array, ddq_all[i])
        axs[2].grid(True)
        axs[3].set_title("Joint Jerk (rad/s³)")
        for i in range(n_joints):
            axs[3].plot(t_array, dddq_all[i])
        axs[3].grid(True)
        axs[3].set_xlabel("time (s)")
        plt.tight_layout()
        plt.show()