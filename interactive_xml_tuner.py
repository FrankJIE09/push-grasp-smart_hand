import mujoco
import mujoco.viewer
import os
import time
from pynput import keyboard

# 定义要加载的XML文件路径
# Define the path to the XML file to be loaded
xml_path = 'push-grasp-scene.xml'


# 这个列表用作一个可变标志，以便在回调函数中修改它，并在主循环中检测到变化
# This list is used as a mutable flag so it can be modified inside the callback
# and the change can be detected in the main loop.
reload_request = [False]
print_camera_request = [False]

def on_press(key):
    """
    pynput 键盘回调函数。
    - 'R'/'r': 设置重载请求标志。
    - 'C'/'c': 设置打印相机视角请求标志。
    pynput key callback function. 
    - 'R'/'r': Sets the reload request flag.
    - 'C'/'c': Sets the print camera info request flag.
    """
    try:
        upper_char = key.char.upper()
        if upper_char == 'R':
            if not reload_request[0]:
                reload_request[0] = True
                print("\n>>>>> 'R' 键被按下。正在准备重新加载模型... <<<<<\n")
        elif upper_char == 'C':
            if not print_camera_request[0]:
                print_camera_request[0] = True
                print("\n>>>>> 'C' 键被按下。正在获取当前相机视角... <<<<<\n")
    except AttributeError:
        # 非字符键（如Shift, Ctrl）会被忽略
        # Non-character keys (like Shift, Ctrl) are ignored
        pass

def load_model_and_data(path):
    """
    从指定路径加载模型。如果XML有语法错误，会捕获异常并返回None。
    Loads a model from the specified path. Returns None if the XML is malformed.
    """
    try:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        return model, data
    except Exception as e:
        print(f"\n!!!!!!!!!!!!!! XML文件加载失败 !!!!!!!!!!!!!!")
        print(f"错误: {e}")
        print(f"请检查并修正 '{path}' 文件后，按 Enter 键重试。")
        return None, None

# =================================================================================
# 主程序循环 - 实现热重载
# Main application loop for hot-reloading
# =================================================================================

# 启动 pynput 键盘监听器
# Start the pynput keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

print("键盘监听器已启动。按 'R' 键可热重载模型，按 'C' 键可获取相机视角。")

try:
    while True:
        reload_request[0] = False  # 在每次循环开始时重置标志 Reset the flag at the start of each loop
        print_camera_request[0] = False # 同上 for camera

        print("=====================================================================")
        print(f"正在从 '{xml_path}' 加载模型...")
        model, data = load_model_and_data(xml_path)

        # 如果模型加载失败，等待用户修复文件后重试
        # If model loading fails, wait for the user to fix the file and retry
        if model is None:
            input()  # 等待用户按 Enter Wait for user to press Enter
            continue

        # 设置重力为0
        # Set gravity to zero
        model.opt.gravity[:] = [0, 0, 0]
        print(f"✅ 重力已设置为: {model.opt.gravity}")

        # 使用 launch_passive 启动查看器
        # Launch the viewer using launch_passive
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # user_scn is for adding custom geoms. Its `scale` attribute might
            # scale these custom geoms, or potentially the entire scene as a
            # viewer-specific feature. We keep this experimental setting.
            viewer.user_scn.scale = 3

            # 设置默认相机视角
            # Set default camera view
            viewer.cam.azimuth = -28.17
            viewer.cam.elevation = -14.98
            viewer.cam.distance = 0.30
            viewer.cam.lookat[:] = [-0.05, 0.03, 1.96]
            
            # 不再需要绑定 MuJoCo 的键盘回调
            # No longer need to set MuJoCo's key callback
            
            print("\n✅ 模型加载成功! 仿真器已启动。")
            print("--------------------------------------------------")
            print("  - 修改XML: 在您的编辑器中修改并保存 ahand_left.xml 文件。")
            print("  - 热重载:   在 *任何窗口* 按下 'R' 键 (无需聚焦于仿真器窗口)。")
            print("  - 获取视角: 在 *任何窗口* 按下 'C' 键，在终端查看当前相机参数。")
            print("  - 控  制:   在仿真器窗口中按 'Ctrl+L' 打开控制滑块。")
            print("  - 字体/UI:  通过 launch_passive 无法编程控制。请在查看器窗口手动调整。")
            print("  - 退  出:   直接关闭仿真器窗口。")
            print("=====================================================================\n")

            # 仿真子循环
            # Inner simulation loop
            while viewer.is_running() and not reload_request[0]:
                step_start = time.time()
                
                # 检查是否需要打印相机信息
                if print_camera_request[0]:
                    print("\n📸 ========== 当前相机视角 ========== 📸")
                    print(f"viewer.cam.azimuth = {viewer.cam.azimuth:.2f}")
                    print(f"viewer.cam.elevation = {viewer.cam.elevation:.2f}")
                    print(f"viewer.cam.distance = {viewer.cam.distance:.2f}")
                    print(f"viewer.cam.lookat[:] = [{viewer.cam.lookat[0]:.2f}, {viewer.cam.lookat[1]:.2f}, {viewer.cam.lookat[2]:.2f}]")
                    print("========================================\n")
                    print_camera_request[0] = False # 重置标志

                mujoco.mj_step(model, data)
                viewer.sync()
                # time.sleep(0.01)
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            
            # 如果循环是因为关闭窗口而结束（而不是因为请求重载）
            # If the loop ended because the window was closed (and not due to a reload request)
            if not viewer.is_running() and not reload_request[0]:
                print("仿真器已关闭。程序退出。")
                break  # 退出主循环 Exit the main while loop

        # 如果代码执行到这里，说明是 reload_request[0] 变为了 True
        # If the code reaches here, it means reload_request[0] became True
        # 主循环将自动继续，从而重新加载模型
        # The main loop will now continue, thereby reloading the model
finally:
    print("正在停止键盘监听器...")
    listener.stop()
    listener.join()
    print("键盘监听器已停止。") 