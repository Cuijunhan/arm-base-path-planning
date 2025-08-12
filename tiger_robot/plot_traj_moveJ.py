# visualize_trajectories.py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

def load_csv(filename):
    return pd.read_csv(filename,skipinitialspace=True)

def quat_to_euler(df):
    r = R.from_quat(df[['qx', 'qy', 'qz', 'qw']].values)
    return r.as_euler('zyx', degrees=True)  # yaw, pitch, roll

# def plot_position(df, label_prefix):
#     plt.figure(figsize=(10, 5))
#     for col, label in zip(['px', 'py', 'pz'], ['X', 'Y', 'Z']):
#         plt.plot(df['step'], df[col], label=f"{label_prefix} {label}")
#     plt.title(f"{label_prefix} 末端位置变化")
#     plt.xlabel("Step")
#     plt.ylabel("位置 (m)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
def plot_position(df, label_prefix):
    # 清理列名
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    # 转为浮点型
    for col in ['step', 'px', 'py', 'pz']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    plt.figure(figsize=(10, 5))
    for col, label in zip(['px', 'py', 'pz'], ['X', 'Y', 'Z']):
        plt.plot(df['step'].to_numpy(), df[col].to_numpy(), label=f"{label_prefix} {label}")
    plt.title(f"{label_prefix} 末端位置变化")
    plt.xlabel("Step")
    plt.ylabel("位置 (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_orientation(euler, df, label_prefix):
    # 清理列名
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
        plt.plot(df['step'].to_numpy(), euler[:, i], label=f"{label_prefix} {label}")
    plt.title(f"{label_prefix} 末端欧拉角变化")
    plt.xlabel("Step")
    plt.ylabel("角度 (rad)")
    plt.legend()
    plt.grid(True)

# def plot_3d_trajectory(moveJ_df, moveL_df):
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # 绘制 moveJ
#     ax.plot(moveJ_df['px'], moveJ_df['py'], moveJ_df['pz'], label='moveJ Path', color='blue')
#     ax.scatter(moveJ_df['px'].iloc[0], moveJ_df['py'].iloc[0], moveJ_df['pz'].iloc[0], color='blue', marker='o', s=50, label='moveJ Start')
#     ax.scatter(moveJ_df['px'].iloc[-1], moveJ_df['py'].iloc[-1], moveJ_df['pz'].iloc[-1], color='blue', marker='x', s=50, label='moveJ End')

#     # 绘制 moveL
#     ax.plot(moveL_df['px'], moveL_df['py'], moveL_df['pz'], label='moveL Path', color='red')
#     ax.scatter(moveL_df['px'].iloc[0], moveL_df['py'].iloc[0], moveL_df['pz'].iloc[0], color='red', marker='o', s=50, label='moveL Start')
#     ax.scatter(moveL_df['px'].iloc[-1], moveL_df['py'].iloc[-1], moveL_df['pz'].iloc[-1], color='red', marker='x', s=50, label='moveL End')

#     ax.set_title("末端执行器 3D 轨迹")
#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
#     ax.set_zlabel("Z (m)")
#     ax.legend()
#     plt.show()
def plot_3d_trajectory(moveJ_df, moveL_df):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 强制转numpy数组
    ax.plot(moveJ_df['px'].to_numpy(), moveJ_df['py'].to_numpy(), moveJ_df['pz'].to_numpy(), label='moveJ Path', color='blue')
    ax.scatter(moveJ_df['px'].to_numpy()[0], moveJ_df['py'].to_numpy()[0], moveJ_df['pz'].to_numpy()[0], color='blue', marker='o', s=50, label='moveJ Start')
    ax.scatter(moveJ_df['px'].to_numpy()[-1], moveJ_df['py'].to_numpy()[-1], moveJ_df['pz'].to_numpy()[-1], color='blue', marker='x', s=50, label='moveJ End')

    ax.plot(moveL_df['px'].to_numpy(), moveL_df['py'].to_numpy(), moveL_df['pz'].to_numpy(), label='moveL Path', color='red')
    ax.scatter(moveL_df['px'].to_numpy()[0], moveL_df['py'].to_numpy()[0], moveL_df['pz'].to_numpy()[0], color='red', marker='o', s=50, label='moveL Start')
    ax.scatter(moveL_df['px'].to_numpy()[-1], moveL_df['py'].to_numpy()[-1], moveL_df['pz'].to_numpy()[-1], color='red', marker='x', s=50, label='moveL End')

    ax.set_title("末端执行器 3D 轨迹")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.show()


def clean_position(df):
    for col in ['px', 'py', 'pz']:
        df[col] = df[col].astype(str).str.replace(' ', '').astype(float)
    return df

if __name__ == "__main__":
    moveJ_df = load_csv("./build/traj_moveJ_endposes.csv")
    moveL_df = load_csv("./build/traj_moveL_endposes.csv")
    print(moveJ_df)

    moveJ_df = clean_position(moveJ_df)
    moveL_df = clean_position(moveL_df)

    print(moveJ_df)


    moveJ_euler = quat_to_euler(moveJ_df)
    moveL_euler = quat_to_euler(moveL_df)

    plot_position(moveJ_df, "moveJ")
    plot_position(moveL_df, "moveL")

    plot_orientation(moveJ_euler, moveJ_df, "moveJ")
    plot_orientation(moveL_euler, moveL_df, "moveL")

    plot_3d_trajectory(moveJ_df, moveL_df)
