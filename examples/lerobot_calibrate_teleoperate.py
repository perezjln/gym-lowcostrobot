
import argparse
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.koch import KochRobot


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM1", help="Port for the leader motors")
    parser.add_argument("--follower-port", type=str, default="/dev/ttyACM0", help="Port for the follower motors")
    parser.add_argument("--calibration-path", type=str, default="koch.pkl", help="Path to the robots calibration file")
    args = parser.parse_args()

    leader = DynamixelMotorsBus(
                port=args.leader_port,
                motors={
                    # name: (index, model)
                    "shoulder_pan": (1, "xl330-m077"),
                    "shoulder_lift": (2, "xl330-m077"),
                    "elbow_flex": (3, "xl330-m077"),
                    "wrist_flex": (4, "xl330-m077"),
                    "wrist_roll": (5, "xl330-m077"),
                    "gripper": (6, "xl330-m077"),
                },
            )

    follower = DynamixelMotorsBus(
                port=args.follower_port,
                motors={
                    # name: (index, model)
                    "shoulder_pan": (1, "xl430-w250"),
                    "shoulder_lift": (2, "xl430-w250"),
                    "elbow_flex": (3, "xl330-m288"),
                    "wrist_flex": (4, "xl330-m288"),
                    "wrist_roll": (5, "xl330-m288"),
                    "gripper": (6, "xl330-m288"),
                },
            )

    robot = KochRobot(leader_arms={"main": leader}, 
                    follower_arms={"main": follower},
                    calibration_path=args.calibration_path)

    robot.connect()

    while True:
        robot.teleop_step()