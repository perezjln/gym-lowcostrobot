"""Wrapper for recording videos."""

import os

import gymnasium as gym
import h5py
import numpy as np
from gymnasium import logger

"""
observations
observations/images
observations/images/front
observations/images/top
observations/qpos
observations/qvel
"""


class HDF5_Recorder:
    def __init__(self):
        self.step_id = 0
        self.terminated = False
        self.truncated = False
        self.recorded_frames = 0
        self.episode_id = 0
        self.hdf5_file = None

        self.lst_observations = []
        self.lst_actions = []

    def start_hdf5_recorder(self, hdf5_file):
        """Starts HDF5 recorder using :class:`HDF5_Recorder`."""
        self.close()
        self.hdf5_file = hdf5_file
        self.recorded_frames = 1
        self.recording = True
        self.episode_id += 1

    def capture_frame(self, observations, action):
        """Captures frame to video."""
        assert self.hdf5_file is not None
        self.lst_observations.append(observations)
        self.lst_actions.append(action)
        self.recorded_frames += 1

    # numpy.stack([item["image_front"] for item in self.lst_observations])

    def close(self):
        """Closes the hdf5 file."""
        if self.hdf5_file is not None:
            with h5py.File(self.hdf5_file, "w") as file:
                file.create_dataset(
                    "observations/images/front", data=np.stack([item["image_front"] for item in self.lst_observations])
                )
                file.create_dataset(
                    "observations/images/top", data=np.stack([item["image_top"] for item in self.lst_observations])
                )
                file.create_dataset("observations/qpos", data=np.stack([item["arm_qpos"] for item in self.lst_observations]))
                file.create_dataset("observations/qvel", data=np.stack([item["arm_qvel"] for item in self.lst_observations]))
                file.create_dataset("action", data=self.lst_actions)
        self.recorded_frames = 1
        self.lst_observations = []
        self.lst_actions = []


class RecordHDF5Wrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        hdf5_folder: str,
        length: int = 0,
        name_prefix: str = "hdf5_record",
        disable_logger: bool = False,
    ):
        gym.Wrapper.__init__(self, env)

        self.hdf5_folder = os.path.abspath(hdf5_folder)
        self.hdf5_recorder = HDF5_Recorder()

        # Create output folder if needed
        if os.path.isdir(self.hdf5_folder):
            logger.warn(
                f"Overwriting existing videos at {self.hdf5_folder} folder "
                f"(try specifying a different `hdf5_folder` for the `RecordHDF5` wrapper if this is not desired)"
            )
        os.makedirs(self.hdf5_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.length = length
        self.terminated = False
        self.episode_id = 0
        self.env = env

        try:
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.is_vector_env = False

    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations, _ = self.env.reset(**kwargs)
        self.terminated = False
        self.start_hdf5_recorder()
        return observations

    def start_hdf5_recorder(self):
        """Starts video recorder using :class:`video_recorder.VideoRecorder`."""
        self.close_hdf5_recorder()

        video_name = f"{self.name_prefix}-episode-{self.episode_id}.hdf5"
        self.hdf5_recorder.start_hdf5_recorder(hdf5_file=os.path.join(self.hdf5_folder, video_name))
        self.recording = True
        self.episode_id += 1

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""

        observations, rewards, terminateds, truncateds, infos = self.env.step(action)

        # increment steps and episodes
        if self.recording:
            assert self.hdf5_recorder is not None

            self.hdf5_recorder.capture_frame(observations, action)

            if self.length > 0:
                if self.hdf5_recorder.recorded_frames > self.length:
                    self.close_hdf5_recorder()
            else:
                if not self.is_vector_env:
                    if terminateds or truncateds:
                        self.start_hdf5_recorder()
                elif terminateds[0] or truncateds[0]:
                    self.start_hdf5_recorder()

        return observations, rewards, terminateds, truncateds, infos

    def close_hdf5_recorder(self):
        """Closes the hdf5 recorder if currently recording."""
        if self.hdf5_recorder is not None:
            self.hdf5_recorder.close()
        self.recorded_frames = 1

    def render(self, *args, **kwargs):
        return super().render(*args, **kwargs)

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        self.close_hdf5_recorder()
