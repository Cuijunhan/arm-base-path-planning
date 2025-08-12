// 文件结构说明：
// 1) main.cpp            -- C++ 源代码（包含 FK、数值 IK、moveJ、moveL、以及
// demo 验证） 2) CMakeLists.txt      -- 用于编译的 CMake 文件 编译方法： mkdir
// build && cd build cmake .. && make 运行： ./move_demo

// ===================== main.cpp =====================

#include "matplotlibcpp.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace plt = matplotlibcpp;

using namespace Eigen;
using namespace std;

struct MDHRow {
  double a;            // link length
  double alpha;        // link twist
  double d;            // link offset
  double theta_offset; // constant offset added to joint angle
};

// 使用 Modified DH (mDH) 的变换：
// A_i = RotZ(theta) * TransZ(d) * TransX(a) * RotX(alpha)
// 请根据你实际的 mDH 定义确认这个实现是否与您的参数定义一致，若不同请调整。

Matrix4d transform_mdh(double theta, const MDHRow &p) {
  double a = p.a;
  double alpha = p.alpha;
  double d = p.d;
  double th = theta + p.theta_offset;

  Matrix4d T = Matrix4d::Identity();
  // RotZ(th)
  Matrix3d Rz;
  Rz = AngleAxisd(th, Vector3d::UnitZ()).toRotationMatrix();
  T.block<3, 3>(0, 0) = Rz;
  T(0, 3) = 0;
  T(1, 3) = 0;
  T(2, 3) = 0;
  // TransZ(d)
  Matrix4d Tz = Matrix4d::Identity();
  Tz(2, 3) = d;
  // TransX(a)
  Matrix4d Tx = Matrix4d::Identity();
  Tx(0, 3) = a;
  // RotX(alpha)
  Matrix4d Rx = Matrix4d::Identity();
  Rx.block<3, 3>(0, 0) =
      AngleAxisd(alpha, Vector3d::UnitX()).toRotationMatrix();

  return T * Tz * Tx * Rx;
}

// Forward Kinematics: 给定 7 个关节角，返回末端 4x4 齐次矩阵
Matrix4d forward_kinematics(const vector<MDHRow> &mdh, const VectorXd &q) {
  Matrix4d T = Matrix4d::Identity();
  for (size_t i = 0; i < mdh.size(); ++i) {
    T = T * transform_mdh(q(i), mdh[i]);
  }
  return T;
}

// 将 4x4 转换为位置 + 四元数
void poseFromMatrix(const Matrix4d &T, Vector3d &pos, Quaterniond &quat) {
  pos = T.block<3, 1>(0, 3);
  Matrix3d R = T.block<3, 3>(0, 0);
  quat = Quaterniond(R);
}

// 计算位姿误差: 6-vector [pos_err; rot_err]，rot_err 使用角轴 small-angle 近似
VectorXd poseError(const Vector3d &p_des, const Quaterniond &q_des,
                   const Vector3d &p_cur, const Quaterniond &q_cur) {
  VectorXd e(6);
  e.segment<3>(0) = p_des - p_cur;
  // 旋转误差： r_err = 0.5 * (R_des * R_cur^T - R_cur * R_des^T) 的反对称部分,
  // 简化使用四元数差
  Quaterniond q_err = q_des * q_cur.conjugate();
  Vector3d axis;
  double angle;
  // 将 q_err 转换为等价的角轴（小角近似也适用）
  angle = 2 * std::acos(std::min(1.0, std::max(-1.0, q_err.w())));
  if (std::abs(angle) < 1e-9)
    axis = Vector3d::Zero();
  else {
    axis = q_err.vec() / std::sin(angle / 2.0);
  }
  e.segment<3>(3) = axis * angle;
  return e;
}

// 数值 Jacobian（6x7）通过微小扰动计算
MatrixXd numericJacobian(const vector<MDHRow> &mdh, const VectorXd &q) {
  const double eps = 1e-8;
  int n = q.size();
  MatrixXd J(6, n);
  Vector3d p0;
  Quaterniond q0;
  Matrix4d T0 = forward_kinematics(mdh, q);
  poseFromMatrix(T0, p0, q0);

  for (int i = 0; i < n; ++i) {
    VectorXd q_pert = q;
    q_pert(i) += eps;
    Matrix4d Ti = forward_kinematics(mdh, q_pert);
    Vector3d pi;
    Quaterniond qi;
    poseFromMatrix(Ti, pi, qi);
    VectorXd err = poseError(pi, qi, p0, q0);
    J.col(i) = err / eps;
  }
  return J;
}

// Damped least squares IK（6x7 -> using numerical Jacobian）
// 返回 (success, q_sol)
pair<bool, VectorXd> numericalIK(const vector<MDHRow> &mdh,
                                 const VectorXd &q_init, const Vector3d &p_des,
                                 const Quaterniond &q_des, int max_iters = 200,
                                 double tol = 1e-4) {
  VectorXd q = q_init;
  double lambda = 1e-2;
  for (int it = 0; it < max_iters; ++it) {
    Matrix4d T = forward_kinematics(mdh, q);
    Vector3d p_cur;
    Quaterniond q_cur;
    poseFromMatrix(T, p_cur, q_cur);
    VectorXd err = poseError(p_des, q_des, p_cur, q_cur);
    double err_norm = err.norm();
    if (err_norm < tol)
      return {true, q};

    MatrixXd J = numericJacobian(mdh, q); // 6x7
    MatrixXd JJt = J * J.transpose();
    MatrixXd inv = (JJt + lambda * lambda * MatrixXd::Identity(6, 6)).inverse();
    VectorXd delta_q = J.transpose() * (inv * err);
    // 步长限制
    double max_step = 0.1; // rad
    if (delta_q.norm() > max_step)
      delta_q *= (max_step / delta_q.norm());

    q += delta_q;
  }
  return {false, q};
}

// moveJ: 关节空间插补（线性插值 + 可选速度约束）
// 返回关节轨迹（每行是一个时间步的关节向量）
// 等时等比例关节插值轨迹
// vector<VectorXd> moveJ_trajectory(const VectorXd &q_start,
//                                   const VectorXd &q_goal, double dt = 0.01,
//                                   double max_joint_speed = 1.0) {
//   int n = q_start.size();
//   double max_delta = 0;
//   for (int i = 0; i < n; ++i)
//     max_delta = max(max_delta, fabs(q_goal(i) - q_start(i)));
//   double T = max_delta / max_joint_speed;
//   if (T < 1e-6)
//     T = 0.0;
//   int steps = (T > 0) ? int(ceil(T / dt)) : 1;
//   vector<VectorXd> traj;
//   for (int s = 0; s <= steps; ++s) {
//     double tau = (steps == 0) ? 1.0 : double(s) / steps;
//     VectorXd q = q_start + tau * (q_goal - q_start);
//     traj.push_back(q);
//   }
//   return traj;
// }

////////////////  moveJ S曲线插补  ////////////////
struct SProfile {
  double q0, qf;
  double vmax, amax, jmax;
  double Tj, Ta, Tv, Td, T;
};

SProfile planSProfile(double q0, double qf, double vmax, double amax,
                      double jmax) {
  SProfile sp;
  sp.q0 = q0;
  sp.qf = qf;
  sp.vmax = vmax;
  sp.amax = amax;
  sp.jmax = jmax;

  double D = qf - q0;
  double dir = (D >= 0) ? 1.0 : -1.0;
  D = fabs(D);

  sp.Tj = amax / jmax;
  double D_min = 2 * ((amax * amax) / (jmax) + (amax * vmax) / (jmax));

  if (D < D_min) {
    // 没有匀速段 -> 三角S曲线
    sp.Tj = cbrt(D / (2 * jmax));
    sp.Ta = 2 * sp.Tj;
    sp.Tv = 0;
  } else {
    // 有匀速段 -> 梯形S曲线
    sp.Ta = amax / jmax + (vmax / amax);
    sp.Tv = (D - 2 * (amax * amax) / (jmax)) / vmax;
  }
  sp.Td = sp.Ta;
  sp.T = sp.Ta + sp.Tv + sp.Td;
  return sp;
}

double sCurvePos(double t, const SProfile &sp) {
  double dir = (sp.qf >= sp.q0) ? 1.0 : -1.0;
  double amax = sp.amax * dir;
  double jmax = sp.jmax * dir;
  double q = sp.q0;

  if (t < sp.Tj) {
    q += jmax / 6 * t * t * t;
  } else if (t < sp.Ta - sp.Tj) {
    q += jmax / 6 * sp.Tj * sp.Tj * sp.Tj +
         0.5 * amax * (t - sp.Tj) * (t - sp.Tj) + amax * sp.Tj * (t - sp.Tj);
  } else if (t < sp.Ta) {
    double dt = t - (sp.Ta - sp.Tj);
    q += jmax / 6 * sp.Tj * sp.Tj * sp.Tj +
         0.5 * amax * (sp.Ta - 2 * sp.Tj) * (sp.Ta - 2 * sp.Tj) +
         amax * sp.Tj * (sp.Ta - 2 * sp.Tj) + amax * (sp.Ta - t) * dt -
         jmax / 6 * dt * dt * dt;
  } else if (t < sp.Ta + sp.Tv) {
    q += (sp.qf - sp.q0 - 0.5 * amax * sp.Ta * sp.Ta -
          0.5 * amax * sp.Td * sp.Td) /
         sp.Tv * (t - sp.Ta);
  } else {
    double td = t - (sp.Ta + sp.Tv);
    q = sp.qf - jmax / 6 * td * td * td; // 对称减速
  }
  return q;
}

// 多关节moveJ
vector<VectorXd> moveJ_Scurve(const VectorXd &q_start, const VectorXd &q_goal,
                              double vmax = 1.0, double amax = 2.0,
                              double jmax = 10.0, double dt = 0.01) {
  int n = q_start.size();
  vector<SProfile> profiles;
  profiles.reserve(n);

  // 先计算每个关节的轨迹参数
  double Tmax = 0.0;
  for (int i = 0; i < n; ++i) {
    auto sp = planSProfile(q_start[i], q_goal[i], vmax, amax, jmax);
    profiles.push_back(sp);
    Tmax = max(Tmax, sp.T);
  }

  // 同步所有关节的时间
  vector<VectorXd> traj;
  int steps = int(ceil(Tmax / dt));
  for (int s = 0; s <= steps; ++s) {
    double t = s * dt;
    VectorXd q(n);
    for (int i = 0; i < n; ++i) {
      double t_scaled = min(t, profiles[i].T);
      q[i] = sCurvePos(t_scaled, profiles[i]);
    }
    traj.push_back(q);
  }
  return traj;
}

///////////////////  moveJ S曲线插补  ////////////////

// slerp for quaternions
Quaterniond slerp_q(const Quaterniond &q1, const Quaterniond &q2, double t) {
  return q1.slerp(t, q2);
}

// moveL: 末端笛卡尔直线插补
// 每一步在末端空间线性插值位置，姿态使用 slerp，然后用 IK 求解关节解
vector<VectorXd> moveL_trajectory(const vector<MDHRow> &mdh,
                                  const VectorXd &q_start,
                                  const Vector3d &p_goal,
                                  const Quaterniond &q_goal,
                                  double step_len = 0.01) {
  // 得到起始位姿
  Matrix4d T0 = forward_kinematics(mdh, q_start);
  Vector3d p0;
  Quaterniond q0;
  poseFromMatrix(T0, p0, q0);
  double total_dist = (p_goal - p0).norm();
  int steps = max(1, int(ceil(total_dist / step_len)));
  vector<VectorXd> traj;
  VectorXd cur_q = q_start;
  for (int i = 0; i <= steps; ++i) {
    double t = double(i) / steps;
    Vector3d p_t = p0 + t * (p_goal - p0);
    Quaterniond q_t = slerp_q(q0, q_goal, t);
    auto res = numericalIK(mdh, cur_q, p_t, q_t, 200, 1e-4);
    if (!res.first) {
      cerr << "Warning: IK failed at step " << i << " t=" << t << "\n";
      // 还是把当前猜测放进去（粗糙处理）
      traj.push_back(cur_q);
    } else {
      cur_q = res.second;
      traj.push_back(cur_q);
    }
  }
  return traj;
}

// 简单验证：对轨迹中的每个关节点计算 FK，输出末端位姿到 CSV
void dump_trajectory_endposes(const vector<MDHRow> &mdh,
                              const vector<VectorXd> &traj,
                              const string &fname) {
  ofstream ofs(fname);
  ofs << "step,px,py,pz,qw,qx,qy,qz"
      << "\n";
  for (size_t i = 0; i < traj.size(); ++i) {
    Matrix4d T = forward_kinematics(mdh, traj[i]);
    Vector3d p;
    Quaterniond q;
    poseFromMatrix(T, p, q);
    ofs << i << "," << p.transpose()[0] << "," << p.transpose()[1] << ","
        << p.transpose()[2] << "," << q.w() << "," << q.x() << "," << q.y()
        << "," << q.z() << "\n";
  }
  ofs.close();
}

int main() {
  // --- 示例 mDH 参数（示例用，请替换为你自己的 mDH 参数）
  vector<MDHRow> mdh(7);
  // 这里给出一个简单可运行的参考链（单位：米 / 弧度）
  mdh[0] = {0.0, -M_PI / 2, 0.34, 0.0};
  mdh[1] = {0.0, M_PI / 2, 0.0, 0.0};
  mdh[2] = {0.0, M_PI / 2, 0.4, 0.0};
  mdh[3] = {0.0, -M_PI / 2, 0.0, 0.0};
  mdh[4] = {0.0, M_PI / 2, 0.39, 0.0};
  mdh[5] = {0.0, -M_PI / 2, 0.0, 0.0};
  mdh[6] = {0.0, 0.0, 0.078, 0.0};

  int n = 7;
  VectorXd q_start(n);
  q_start.setZero();
  VectorXd q_goal(n);
  q_goal << 0.2, -0.5, 0.3, -1.0, 0.8, 0.5, -0.2;

  cout << "=== moveJ 演示 ===\n";
  //   auto trajJ = moveJ_trajectory(q_start, q_goal, 0.02, 0.6);
  auto trajJ = moveJ_Scurve(q_start, q_goal, 1.0, 2.0, 10.0, 0.01);

  dump_trajectory_endposes(mdh, trajJ, "traj_moveJ_endposes.csv");
  cout << "moveJ 轨迹点数: " << trajJ.size()
       << " 已写入 traj_moveJ_endposes.csv\n";

  // 验证末端是否到达期望（用 FK）
  Matrix4d T_goal = forward_kinematics(mdh, q_goal);
  Vector3d p_goal;
  Quaterniond quat_goal;
  poseFromMatrix(T_goal, p_goal, quat_goal);
  cout << "goal end effector pose from q_goal: pos=" << p_goal.transpose()
       << " quat=" << quat_goal.coeffs().transpose() << "\n";

  cout << "=== moveL 演示 ===\n";
  // 目标末端位姿：在当前末端基础上平移 0.2m
  Matrix4d T0 = forward_kinematics(mdh, q_start);
  Vector3d p0;
  Quaterniond q0;
  poseFromMatrix(T0, p0, q0);
  Vector3d p_des = p0 + Vector3d(0.2, 0.0, -0.05);
  Quaterniond q_des = q0; // 保持姿态不变

  auto trajL = moveL_trajectory(mdh, q_start, p_des, q_des, 0.02);
  dump_trajectory_endposes(mdh, trajL, "traj_moveL_endposes.csv");
  cout << "moveL 轨迹点数: " << trajL.size()
       << " 已写入 traj_moveL_endposes.csv\n";

  // 检查最终末端误差
  Matrix4d T_last = forward_kinematics(mdh, trajL.back());
  Vector3d p_last;
  Quaterniond q_last;
  poseFromMatrix(T_last, p_last, q_last);
  cout << "moveL final pos=" << p_last.transpose()
       << " desired pos=" << p_des.transpose()
       << " error=" << (p_des - p_last).norm() << "\n";

  cout << "运行结束。请查看生成的 CSV 文件 (traj_moveJ_endposes.csv / "
          "traj_moveL_endposes.csv)\n";
  cout << "建议使用 Python 或 Excel 打开 CSV 可视化位姿随时间的变化（或在 RViz "
          "/ Meshcat 中可视化）\n";

  return 0;
}

// ===================== CMakeLists.txt =====================
//
// cmake_minimum_required(VERSION 3.10)
// project(move_demo)
// set(CMAKE_CXX_STANDARD 17)
// find_package(Eigen3 REQUIRED)
// add_executable(move_demo main.cpp)
// target_include_directories(move_demo PRIVATE ${EIGEN3_INCLUDE_DIR})

// 如果 find_package(Eigen3) 无法找到，请安装 libeigen3-dev 或在 CMake
// 中手动指定 Eigen 路径。
