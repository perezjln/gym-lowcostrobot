import time
import numpy as np
import threading
import mujoco.viewer

from gym_lowcostrobot.interface.dynamixel import Dynamixel
from gym_lowcostrobot.interface.robot import Robot
from gym_lowcostrobot.interface.simulated_robot import SimulatedRobot

def read_leader_position():
    global target_pos
    while True:
        target_pos = np.array(leader.read_position())
        target_pos = (target_pos / 2048 - 1) * 3.14
        target_pos[1] = -target_pos[1]
        target_pos[3] = -target_pos[3]
        target_pos[4] = -target_pos[4]

if __name__ == '__main__':

    leader_dynamixel = Dynamixel.Config(baudrate=1_000_000, device_name='/dev/tty.usbmodem57380045631').instantiate()
    leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 6, 7])
    leader.set_trigger_torque()

    m = mujoco.MjModel.from_xml_path('gym_lowcostrobot/assets/low_cost_robot_6dof/reach_cube.xml')
    d = mujoco.MjData(m)
    r = SimulatedRobot(m, d)

    target_pos = np.zeros(5)

    # Start the thread for reading leader position
    leader_thread = threading.Thread(target=read_leader_position)
    leader_thread.daemon = True
    leader_thread.start()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running():
            # Use the latest target_pos
            step_start = time.time()
            target_pos_local = target_pos.copy()
            # print(f'target pos copy {time.time() - step_start}')
            r.set_target_pos(target_pos_local)
            # print(f'set targtee pos copy {time.time() - step_start}')
            mujoco.mj_step(m, d)
            # print(f'mjstep {time.time() - step_start}')
            viewer.sync()
            # print(f'viewer sync {time.time() - step_start}')

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            # print(f'time until next step {time_until_next_step}')
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)