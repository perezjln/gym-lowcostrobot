
import h5py, tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


# Specify the path to your HDF5 file
file_path = 'data/episode_46.hdf5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as file:

    # Access the datasets or attributes in the file
    # For example, to access a dataset named 'data':
    group_obs_img = file['observations/images/front']

    img = [] # some array of images
    frames = [] # for storing the generated images

    # Write each image to the video
    fig = plt.figure()
    for i in tqdm.tqdm(range(group_obs_img.shape[0]), desc='Writing video'):
        frames.append([plt.imshow(group_obs_img[i],animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save('movie.mp4')
