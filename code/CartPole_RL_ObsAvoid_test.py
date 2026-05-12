# -*- coding: utf-8 -*-
"""
DDPG CartPole 避障测试脚本 - 验证起摆与避障效果
"""
import mujoco
import mujoco.viewer
import torch
import numpy as np
import math
import time
import matplotlib.pyplot as plt

max_action = 12.0

# --- 1. 网络架构 (必须与训练时一致：state_dim=6) ---
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

# --- 2. 环境包装 (必须同步 6 维观测逻辑) ---
class CartPoleRL:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.action_space_high = max_action
        self.obs_direction = 1.0 # 障碍物巡逻方向

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] = 0.0 # 车中心
        self.data.qpos[0] = -0.5 # 障碍物初始位置
        self.data.qpos[2] = np.pi # 摆杆向下
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        # 索引同步：0:车, 1:障碍物, 2:摆杆
        x = self.data.qpos[1]
        v = self.data.qvel[1]
        obs_x = self.data.qpos[0]
        obs_v = self.data.qvel[0]
        theta = math.atan2(math.sin(self.data.qpos[2]), math.cos(self.data.qpos[2]))
        omega = self.data.qvel[2]
        
        # 返回 6 维观测 [x, v, theta, omega, obs_x, rel_dist]
        return np.array([
            x, v, theta, omega, obs_v, (x - obs_x)
        ], dtype=np.float32)

    def step(self, action):
        # 1. 施加小车控制力
        self.data.ctrl[0] = np.clip(action, -self.action_space_high, self.action_space_high)
        
        # 2. 同步障碍物巡逻逻辑 (测试时障碍物也得动，才能测试避障)
        if self.data.qpos[0] < -0.8: self.obs_direction = 1.0
        elif self.data.qpos[0] > 0.8: self.obs_direction = -1.0
        self.data.ctrl[1] = self.obs_direction * 0.5
        
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        
        # 结束条件：出界 或 碰撞
        done = abs(self.data.qpos[1]) >= 1.0 or self.data.ncon > 0
        return obs, done

# --- 3. 测试主逻辑 ---
def test():
    # 数据存储
    history = {
        "time": [], "cart_pos": [], "cart_vel": [], 
        "pole_angle": [], "obs_pos": [], "ctrl": []
    }

    model_path = "../obs_cartpole.xml"
    weight_path = "../model/best_actor_obs.pth" # 注意文件名同步
    state_dim = 6  # 关键修改：改为 6
    action_dim = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CartPoleRL(model_path)
    
    actor = Actor(state_dim, action_dim, max_action).to(device)
    try:
        actor.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ 成功加载避障模型: {weight_path}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    actor.eval()
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        start_real_time = time.time()
        state = env.reset()
        
        print("🚀 仿真开始（带障碍物自动巡逻）...")
        while viewer.is_running():
            step_start = time.time()
            
            # 20秒限时
            # if time.time() - start_real_time > 20.0: break
            
            # 预测动作
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().flatten()
            
            # 环境演进
            state, done = env.step(action)
            
            # 记录数据 (state索引: 0:x, 4:obs_x, 2:theta)
            history["time"].append(env.data.time)
            history["cart_pos"].append(state[0])
            history["cart_vel"].append(state[1])
            history["pole_angle"].append(state[2] * 180 / np.pi)
            history["obs_pos"].append(state[4])
            history["ctrl"].append(float(action[0]))

            if done:
                reason = "碰撞" if env.data.ncon > 0 else "出界"
                print(f"终止原因: {reason}")
                break
            
            viewer.sync()
            # 保持频率
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    return history

if __name__ == "__main__":
    h = test()
    
    # --- 绘图 ---
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # 1. 位置对比图 (小车 vs 障碍物)
    axs[0].plot(h["time"], h["cart_pos"], label='Cart', color='blue')
    axs[0].plot(h["time"], h["obs_pos"], label='Obstacle', color='orange', linestyle='--')
    axs[0].set_ylabel('Position (m)')
    axs[0].legend()
    axs[0].grid(True)

    # 2. 摆杆角度
    axs[1].plot(h["time"], h["pole_angle"], color='red')
    axs[1].axhline(y=0, color='black', linestyle='-') # 0度线（目标）
    axs[1].set_ylabel('Pole Angle (deg)')
    axs[1].grid(True)

    # 3. 控制力
    axs[2].plot(h["time"], h["ctrl"], color='black')
    axs[2].set_ylabel('Control Force')
    axs[2].grid(True)

    # 4. 距离 (安全性分析)
    rel_dist = np.array(h["cart_pos"]) - np.array(h["obs_pos"])
    axs[3].plot(h["time"], np.abs(rel_dist), color='green')
    axs[3].axhline(y=0.15, color='r', linestyle=':', label='Danger Zone')
    axs[3].set_ylabel('Relative Dist (m)')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()