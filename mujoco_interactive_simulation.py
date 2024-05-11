import time
import mujoco
import mujoco.viewer

from envs.SimulatedRobot import SimulatedRobot


if __name__ == '__main__':

  m = mujoco.MjModel.from_xml_path('low_cost_robot/scene_one_cube.xml')
  d = mujoco.MjData(m)
  r = SimulatedRobot(m, d)

  with mujoco.viewer.launch_passive(m, d) as viewer:

    start = time.time()

    while viewer.is_running():

      step_start = time.time()
      mujoco.mj_step(m, d)
      viewer.sync()

      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = m.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)