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

xml = f"""
<mujoco model="cartpole">

  <option timestep="0.01" integrator="RK4"/>
  
  <!-- 1. 调整视觉参数，增加环境光（让阴影不那么黑） -->
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
  </visual>
  
  <worldbody>
    <!-- 2. 主光源：位置调高，增加平行光感 -->
    <light name="top_light" pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9" castshadow="true"/>
    <!-- 3. 辅助光源：从侧前方照射，补全阴影细节 -->
    <light name="side_light" pos="2 2 2" dir="-1 -1 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    
    <geom name="floor" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    
    <!-- ======= 轨道系统 ======= -->
    <!-- 轨道主体：位于 z=0.8，长度覆盖 -3 到 3 -->
    <geom name="rail" type="capsule" fromto="-1.5 0 0.73 1.5 0 0.73" size="0.02" rgba="0.3 0.3 0.3 1"/>
    <!-- 左侧支柱 -->
    <geom name="left_support" type="cylinder" fromto="-1.5 0 0 -1.5 0 0.73" size="0.05" rgba="0.5 0.5 0.5 1"/>
    <!-- 右侧支柱 -->
    <geom name="right_support" type="cylinder" fromto="1.5 0 0 1.5 0 0.73" size="0.05" rgba="0.5 0.5 0.5 1"/>
    <!-- ============================ -->
    
    <!--============小车=========-->
    <body name="cart" pos="0 0 0.8">
      <inertial pos="0 0 0" mass="{Mc}" diaginertia="1e-8 1e-8 1e-8"/> <!-- 保持惯量恒定 -->
      <joint name="slider" type="slide" damping="{Beq}" range="-1.0 1.0" axis="1 0 0" />
      <geom name="cart_geom" type="box" size="0.1 0.05 0.05" rgba="0.2 0.6 0.8 1"/>
      
      <!--===========摆杆=================-->
      <body name="pole" pos="0 -0.05 0">
        <!-- pos 是质心相对于 body 原点的位置 -->
        <inertial pos="0 0 {lp}" mass="{Mp}" diaginertia="1e-8 {Ip} {Ip}"/>
        <joint name="hinge" type="hinge" damping="{Bp}" axis="0 1 0" />
        <geom name="pole_geom" type="capsule" size="0.01 0.3" fromto="0 0 0 0 0 {Lp}" rgba="0.8 0.2 0.2 1"/>
      </body>
    </body>
  </worldbody>
  
  <!-- 添加关键帧定义 -->
  <keyframe>
    <!-- name可以自定义，qpos 对应 [小车位置, 摆杆角度] -->
    <!-- 这里的 0.5 表示我们希望重置后摆杆处于 0.5 弧度的位置 -->
    <key name="home" qpos="0 0.15" qvel="0 0"/>
  </keyframe>
  
  <actuator>
    <motor name="motor" joint="slider" gear="1"/>
  </actuator>
  
</mujoco>
"""

# 两种调用模型方式
# 1、在该文件中进行设计
# model = mujoco.MjModel.from_xml_string(xml)
# data = mujoco.MjData(model)

# 2、模型保存为单独的xml文件，然后导入
model = mujoco.MjModel.from_xml_path("cartpole.xml")
data = mujoco.MjData(model)

error_cart = 0
error_pole = 0
print(type(data))

# 初始化数据存储列表
time_history = []
cart_pos_history = []
cart_vel_history = []
pole_angle_history = []
pole_vel_history = []
f_history = []

# 初始化 PID 积分项
error_cart = 0.0
error_pole = 0.0

last_error_cart = 0.0
last_error_pole = 0.0

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

data.qpos[0] = 0.0 
data.qpos[1] = 0.1

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 记录仿真开始时的物理时间
    start_sim_time = data.time 
    
    while viewer.is_running():
        step_start = time.time()
        
        # 1. 检查运行时间限制 (20秒)
        if data.time - start_sim_time > 20.0:
            print("🕒 已达到20秒运行时间，仿真结束。")
            break
        # Mujoco中点击Reset回将data.time置为0
        if data.time == 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            error_cart = 0.0
            error_pole = 0.0
            
            last_error_cart = 0.0
            last_error_pole = 0.0
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
        # --- 控制逻辑 ---
        target_pos = 0.0
        target_angle = 0.0
        
        error1 = cart_pos - target_pos
        error2 = pole_angle - target_angle
        
        # 只有在未倒下的情况下才累加积分，防止离散误差爆炸
        if abs(pole_angle) < 30 * np.pi / 180:
            error_cart += error1
            error_pole += error2
            # 全状态反馈控制
            control_cart = kp_cart * error1 + ki_cart * error_cart * 0.01 + kd_cart * (error1 - last_error_cart)/0.01
            control_pole = kp_pole * error2 + ki_pole * error_pole * 0.01 + kd_pole * (error2 - last_error_pole)/0.01
           
            control = control_pole + control_cart
        else:
            control = 0.0 # 倒下后放弃抵抗
      
        f_history.append(control)
            
        last_error_cart = error1
        last_error_pole = error2
        
        data.ctrl[0] = control
        print(f"当前控制力：{control}")
        # 2. 物理步进
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