import argparse

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tqdm

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trace video from HDF5 trace file")
    parser.add_argument("--file_path", type=str, default="data/reach-episode-0.hdf5", help="Path to HDF5 file")
    parser.add_argument("--camera_id", type=str, default="observations/images/front", help="Camera ID")
    parser.add_argument("--save_file", type=str, default="movie.mp4", help="Path to save the video file")
    args = parser.parse_args()

    # Access the specified HDF5 file
    with h5py.File(args.file_path, "r") as file:
        # Access the datasets or attributes in the file
        group_obs_img = file[args.camera_id]

        img = []  # some array of images
        frames = []  # for storing the generated images

        # Write each image to the video
        fig = plt.figure()
        for i in tqdm.tqdm(range(group_obs_img.shape[0]), desc="Writing video"):
            frames.append([plt.imshow(group_obs_img[i], animated=True)])

        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
        writer = animation.FFMpegWriter(fps=30, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(args.save_file, writer=writer)
