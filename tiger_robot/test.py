import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class DualArmKinematics:
    def __init__(self, urdf_path: str, left_ee_link: str, right_ee_link: str):
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs=[])
        self.model = self.robot.model
        self.data = self.robot.data

        self.left_ee_frame_id = self.model.getFrameId(left_ee_link)
        self.right_ee_frame_id = self.model.getFrameId(right_ee_link)

        # 可视化
        self.visual_model = self.robot.visual_model
        self.visual_data = self.robot.visual_data
        self.viz = MeshcatVisualizer(self.model, self.robot.collision_model, self.visual_model)
        self.viz.initViewer()
        self.viz.loadViewerModel()

    def forward_kinematics(self, q, arm='left'):
        self.robot.forwardKinematics(q)
        self.robot.updateFramePlacements(q)
        if arm == 'left':
            return self.robot.framePlacement(q, self.left_ee_frame_id)
        elif arm == 'right':
            return self.robot.framePlacement(q, self.right_ee_frame_id)
        else:
            raise ValueError("arm must be 'left' or 'right'")

    def inverse_kinematics(self, target_pose, arm='left', q_init=None, max_iter=100, tol=1e-4):
        if q_init is None:
            q = np.zeros(self.model.nq)
        else:
            q = q_init.copy()

        frame_id = self.left_ee_frame_id if arm == 'left' else self.right_ee_frame_id

        for _ in range(max_iter):
            self.robot.forwardKinematics(q)
            self.robot.updateFramePlacements(q)
            current_pose = self.robot.framePlacement(q, frame_id)

            dR = target_pose.rotation @ current_pose.rotation.T
            dtheta = pin.log3(dR)
            dp = target_pose.translation - current_pose.translation

            err = np.concatenate([dp, dtheta])
            if np.linalg.norm(err) < tol:
                return q

            J = self.robot.computeFrameJacobian(q, frame_id, pin.LOCAL)
            dq = np.linalg.pinv(J) @ err
            q += dq

        raise RuntimeError("IK did not converge")

    def moveJ(self, q_start, q_target, steps=100):
        q_start = np.array(q_start)
        q_target = np.array(q_target)
        return [(1 - t) * q_start + t * q_target for t in np.linspace(0, 1, steps)]

    def interpolate_cartesian(self, T_start, T_target, steps=100):
        positions = np.linspace(T_start.translation, T_target.translation, steps)
        rot_start = R.from_matrix(T_start.rotation)
        rot_target = R.from_matrix(T_target.rotation)
        slerped = R.slerp(0, 1, [rot_start, rot_target])(np.linspace(0, 1, steps))
        return [pin.SE3(Ri.as_matrix(), pi) for Ri, pi in zip(slerped, positions)]

    def moveL(self, q_start, T_target, arm='left', steps=100):
        T_start = self.forward_kinematics(q_start, arm=arm)
        interpolated_poses = self.interpolate_cartesian(T_start, T_target, steps)

        traj = []
        q = np.array(q_start)
        for T in interpolated_poses:
            q = self.inverse_kinematics(T, arm=arm, q_init=q)
            traj.append(q.copy())
        return traj

    def display_trajectory(self, trajectory, sleep_time=0.01):
        import time
        for q in trajectory:
            self.viz.display(q)
            time.sleep(sleep_time)

    def neutral(self):
        return pin.neutral(self.model)

    def display_neutral_pose(self):
        q0 = self.neutral()
        self.viz.display(q0)

# 使用示例：
if __name__ == '__main__':
    urdf_path = "/home/cjh/robot_oa/urdf/double_arm.urdf"  # 替换为你的URDF文件路径
    left_ee = "left_ee_link"                # 替换为左臂末端 link 名称
    right_ee = "right_ee_link"              # 替换为右臂末端 link 名称

    arm = DualArmKinematics(urdf_path, left_ee, right_ee)

    arm.display_neutral_pose()


#     q0 = arm.neutral()
#     q1 = np.array(q0) + 0.1

#     # moveJ 左臂演示
#     traj_j = arm.moveJ(q0, q1, steps=50)
#     arm.display_trajectory(traj_j)

#     # # moveL 右臂演示
#     # T_target = pin.SE3(pin.utils.rpyToMatrix(0, 0, np.pi/4), np.array([0.3, -0.2, 0.4]))
#     # traj_l = arm.moveL(q0, T_target, arm='right', steps=50)
#     # arm.display_trajectory(traj_l)

#     print("Joint Trajectory:", traj_j)
#     # 绘制关节轨迹
#     traj_j_np = np.array(traj_j)  # shape: (steps, n_joints)
#     plt.figure()
#     for i in range(traj_j_np.shape[1]):
#         plt.plot(traj_j_np[:, i], label=f'Joint {i+1}')
#     plt.xlabel('Step')
#     plt.ylabel('Joint Value')
#     plt.title('Joint Trajectory')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     T_target = arm.forward_kinematics(q1)
#     q = arm.inverse_kinematics(T_target, q_init=q0)

#     print("Target Pose:", T_target)
#     print("Inverse Kinematics Solution:", q)

# # print("Inverse Kinematics Result:")


#     # # moveL 演示
#     # T_target = arm.forward_kinematics(q1)#pin.SE3(pin.utils.rpyToMatrix(0, 0, np.pi/4), np.array([0.4, 0.2, 0.3]))
#     # traj_l = arm.moveL(q0, T_target, steps=50)
#     # arm.display_trajectory(traj_l)
