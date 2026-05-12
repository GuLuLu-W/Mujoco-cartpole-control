# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:36:50 2026

@author: Gululu
"""

# -*- coding: utf-8 -*-
"""
MuJoCo 模型可视化测试 - 用于检查 XML 搭建效果
"""
import mujoco
import mujoco.viewer
import time
import os
import numpy as np

# 1. 自动处理路径：假设此脚本在 code 文件夹下
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "obs_cartpole.xml")

def visualize():
    # 加载模型
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print(f"✅ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 启动可视化窗口
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print("💡 提示：你可以用鼠标左键拖拽摆杆，右键旋转视角。")
     
        data.ctrl[1] = 10.0
        direction = 1.0
        while viewer.is_running():
            step_start = time.time()
            if data.time == 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                mujoco.mj_forward(model, data)

            # --- 自动测试逻辑：障碍物左右往复运动 ---
            if model.nu > 1:
                # 获取障碍物当前的位置
                obs_pos = data.qpos[0]
                
                # 触碰边界即切换方向
                if obs_pos < -0.8:
                    direction = 1.0
                elif obs_pos > 0.8:
                    direction = -1.0
                    
                # 根据方向施加恒定的力
                data.ctrl[1] = direction * 1.0

            # 物理步进
            mujoco.mj_step(model, data)

            # 同步渲染
            viewer.sync()

            # 保持实时率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    visualize()