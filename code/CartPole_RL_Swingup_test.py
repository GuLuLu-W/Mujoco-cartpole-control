# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:47:09 2026

@author: Gululu
"""

# -*- coding: utf-8 -*-
"""
DDPG CartPole 测试脚本 - 见证摆杆起摆与平衡
"""
import mujoco
import mujoco.viewer
import torch
import numpy as np
import math
import time
import matplotlib.pyplot as plt

max_action = 30.0
# --- 1. 必须导入与训练时完全一致的网络架构 ---
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = torch.nn.Linear(state_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# --- 2. 环境包装 (保持与训练一致的观测逻辑) ---
class CartPoleRL:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.action_space_high = max_action

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # 初始状态：垂直向下
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = np.pi 
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        
        pole_angle = math.atan2(math.sin(self.data.qpos[1]), math.cos(self.data.qpos[1]))
        return np.array([
            self.data.qpos[0], self.data.qvel[0],
            pole_angle, self.data.qvel[1]
        ], dtype=np.float32)

    def step(self, action):
        self.data.ctrl[0] = np.clip(action, -self.action_space_high, self.action_space_high)
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        
        # 定义出界条件 (与训练时保持一致)
        done = abs(self.data.qpos[0]) >= 1.0
        return obs, done

# --- 3. 测试主逻辑 ---
def test():
    
    # 初始化数据存储列表
    time_history = []
    cart_pos_history = []
    cart_vel_history = []
    pole_angle_history = []
    pole_vel_history = []
    f_history = []


    # 参数设置 (需与训练时一致)
    model_path = "../cartpole.xml"
    weight_path = "../model/best_actor.pth"
    state_dim = 4
    action_dim = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化环境
    env = CartPoleRL(model_path)
    
    # 初始化并加载 Actor
    actor = Actor(state_dim, action_dim, max_action).to(device)
    try:
        actor.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"成功加载模型权重: {weight_path}")
    except FileNotFoundError:
        print(f"错误：找不到权重文件 {weight_path}，请确保它在当前目录下。")
        return

    actor.eval() # 切换到评估模式
    
    # 启动渲染器
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        start_real_time = time.time()  # 记录现实开始时间
        print("仿真开始！")
        state = env.reset()
        
        time_history.append(env.data.time)
        cart_pos_history.append(state[0])
        cart_vel_history.append(state[1])
        pole_angle_history.append(state[2] * 180 / np.pi) # 存角度(deg)更直观
        pole_vel_history.append(state[3])

        print(f"初始状态：x:{state[0]};角度:{state[2]*180/np.pi};速度:{state[1]};角速度:{state[3]}")
        while viewer.is_running():
            step_start = time.time()
            
            # 1. 检查运行时间 (现实时间)
            elapsed_time = time.time() - start_real_time
            if elapsed_time > 3.0:
                print(f"🕒 已达到 20 秒限时，仿真结束。")
                break
            # 1. 网络预测动作 (无噪声)
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().flatten()
            
            f_history.append(float(action[0]))
            # 2. 环境步进
            # 3. 物理步进并检查“出界”
            state, done = env.step(action)   
            
            if done:
                print(f"❌ 小车出界 (x={state[0]:.2f})，仿真停止。")
                break
            
            print(f"当前状态：x:{state[0]};角度:{state[2]*180/np.pi};速度:{state[1]};角速度:{state[3]}")
            # 3. 检查是否需要复位 (例如车跑太远)
            if abs(state[0]) >= 1.0:
                print("车跑偏了，正在重置...")
                state = env.reset()
                
            time_history.append(env.data.time)
            cart_pos_history.append(state[0])
            cart_vel_history.append(state[1])
            pole_angle_history.append(state[2] * 180 / np.pi) # 存角度(deg)更直观
            pole_vel_history.append(state[3])

            # 4. 渲染同步
            viewer.sync()

            # 5. 控制物理频率 (保持与 MuJoCo timestep 一致)
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    # --- 修改点 4: 确保控制力列表长度与时间轴对齐 ---
    # 因为 action 是在循环内产生的，f_history 会比 time_history 少一个点
    # 可以在开头补一个 0，或者在绘图时处理
    if len(f_history) < len(time_history):
        f_history.insert(0, 0.0)
        
    return time_history,cart_pos_history,cart_vel_history,pole_angle_history,pole_vel_history,f_history
if __name__ == "__main__":
    time_history,cart_pos_history,cart_vel_history,pole_angle_history,pole_vel_history,f_history = test()
    
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
    