# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:17:49 2026

@author: Gululu
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
import math
import matplotlib.pyplot as plt

Mp = 0.127        # 摆杆质量
Mc=0.38 + 0.37    # 小车质量
lp=0.1778        # 重心到转轴长度
Lp=0.3365         # 摆杆总长度

# 阻尼等参数
Ip=1.2e-03        # 转动惯量
Bp=0.0024         # 摆杆阻尼系数
Beq=5.4          # 小车阻尼系数
# 无阻尼等参数
# Ip=1e-8       # 转动惯量
# Bp=0.0        # 摆杆阻尼系
# Beq=0.0         # 小车阻尼系数

g=9.81           # 重力加速度

# 合并后的转动惯量 (绕支点)
Jp = Ip + Mp * (lp**2) 

# 控制增益
ke = 10.5           # 起摆能量增益 (增大此值起摆更快)
kv_cart = 10.0     # 起摆时的水平阻尼，防止小车跑太远
angle_threshold = 20 * np.pi / 180  # 切换到 PID 的角度阈值 (约20度)

model = mujoco.MjModel.from_xml_path("../cartpole.xml")
data = mujoco.MjData(model)

print(type(data))

# 初始化数据存储列表
time_history = []
cart_pos_history = []
cart_vel_history = []
pole_angle_history = []
pole_vel_history = []
f_history = []

# 无阻尼
# kp_cart = -2.3    
# ki_cart = 5.0
# kd_cart = -1.8

# kp_pole = 70.0    
# ki_pole = 20.0
# kd_pole = 1.0     

# 含阻尼
kp_cart = 250.0    
ki_cart = -12.0
kd_cart = 13.0     

kp_pole = 300.0    
ki_pole = 20.0
kd_pole = 5.0


# 初始化状态：垂直向下 (np.pi)
data.qpos[0] = 0.0 
data.qpos[1] = np.pi 
data.qvel[0] = 1.0

mujoco.mj_forward(model, data)

# PID 累积项初始化
error_cart = 0.0
error_pole = 0.0
last_error_cart = 0.0
last_error_pole = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_sim_time = data.time 
    
    while viewer.is_running():
        step_start = time.time()
        if data.time == 0.0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
        # 1. 检查运行时间限制 (20秒)
        if data.time - start_sim_time > 20.0:
            print("🕒 已达到20秒运行时间，仿真结束。")
            break
        
        # --- 状态获取与归一化 ---
        cart_pos = data.qpos[0]
        pole_angle_raw = data.qpos[1]
        cart_vel = data.qvel[0]
        pole_vel = data.qvel[1]
        
        # 归一化角度至 (-pi, pi)
        pole_angle = math.atan2(math.sin(pole_angle_raw), math.cos(pole_angle_raw))
        
        # --- 记录当前状态数据 ---
        time_history.append(data.time)
        cart_pos_history.append(cart_pos)
        cart_vel_history.append(cart_vel)
        pole_angle_history.append(pole_angle * 180 / np.pi) # 存角度(deg)更直观
        pole_vel_history.append(pole_vel)
        
        print(f"当前时刻状态x:{cart_pos};角度:{pole_angle* 180 / np.pi};速度:{cart_vel};角速度:{pole_vel}")
        
        # --- 控制逻辑切换 ---
        if abs(pole_angle) > angle_threshold:
            # 【阶段一：起摆控制 - 能量法】
            # 目标能量：摆杆在顶端时的势能
            E_target = Mp * g * lp
            # 当前能量：动能 + 势能 (以向上为势能零点，则向下时势能为负)
            E_current = 0.5 * Jp * (pole_vel**2) + Mp * g * lp * (np.cos(pole_angle) - 1)
            
            # 能量误差
            energy_error = E_current - 0 # 这里的 0 是指相对于顶端静止状态的能量差
            
            # 起摆加速度指令 (经典能量控制律)
            # 逻辑：根据能量缺口，在合适的时机推一把小车
            acc = ke * (E_current - E_target) * pole_vel * np.cos(pole_angle)
            
            # 转换为力，并加上简单的水平阻尼防止小车飞出去
            control = (Mc + Mp) * acc - kv_cart * cart_vel
            
            # 切换前清空 PID 积分，防止“积分饱和”
            error_cart = 0.0
            error_pole = 0.0
        else:
            # 【阶段二：稳摆控制 - PID】
            target_pos = 0.0
            target_angle = 0.0
            
            err1 = cart_pos - target_pos
            err2 = pole_angle - target_angle
            
            error_cart += err1
            error_pole += err2
            
            control_cart = kp_cart * err1 + ki_cart * error_cart * 0.01 + kd_cart * (err1 - last_error_cart)/0.01
            control_pole = kp_pole * err2 + ki_pole * error_pole * 0.01 + kd_pole * (err2 - last_error_pole)/0.01
            
            control = control_pole + control_cart
            
            
            last_error_cart = err1
            last_error_pole = err2
        
        f_history.append(control)
        
        # 限幅与执行
        control = np.clip(control, -100, 100)
        data.ctrl[0] = control
        
        print(f"当前控制力：{control}")
        # 物理步进
        mujoco.mj_step(model, data)
        viewer.sync()

        # 保持实时率
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
# --- 仿真结束后绘图 ---
fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
fig.suptitle('Cartpole Simulation States (20s)', fontsize=16)

# 小车位置
axs[0].plot(time_history, cart_pos_history, color='blue')
axs[0].set_ylabel('Cart Pos (m)')
axs[0].grid(True)

# 小车速度
axs[1].plot(time_history, cart_vel_history, color='cyan')
axs[1].set_ylabel('Cart Vel (m/s)')
axs[1].grid(True)

# 摆杆角度
axs[2].plot(time_history, pole_angle_history, color='red')
axs[2].axhline(y=30, color='gray', linestyle='--') # 标记 30度阈值
axs[2].axhline(y=-30, color='gray', linestyle='--')
axs[2].set_ylabel('Pole Angle (deg)')
axs[2].grid(True)

# 摆杆角速度
axs[3].plot(time_history, pole_vel_history, color='magenta')
axs[3].set_ylabel('Pole AngVel (rad/s)')
axs[3].set_xlabel('Time (s)')
axs[3].grid(True)

axs[4].plot(time_history, f_history, color='black')
axs[4].set_ylabel('f')
axs[4].set_xlabel('Time (s)')
axs[4].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()