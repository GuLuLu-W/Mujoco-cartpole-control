Mp = 0.127,        # <--- 注意末尾的逗号
lp = 0.1778,       # <--- 逗号
...
Beq = 5.4,         # <--- 逗号
在 Python 中，如果在数字后面加一个逗号，这个变量就会被识别为一个元组 (Tuple)，而不是一个浮点数 (Float)。

Mujoco中Reset，不会将状态重置回.py代码中设置的初始状态，而是Mujoco自定义的初始状态，或者添加关键帧，可以回到关键帧位置
  <!-- 添加关键帧定义 -->
  <keyframe>
    <!-- name可以自定义，qpos 对应 [小车位置, 摆杆角度] -->
    <!-- 这里的 0.5 表示我们希望重置后摆杆处于 0.5 弧度的位置 -->
    <key name="home" qpos="0 0.15" qvel="0 0"/>
  </keyframe>
  
  调用：mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)