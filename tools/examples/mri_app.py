# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Prepare data
# %% [markdown]
# Let's begin by creating an example image. We will use the Shepp-Logan phantom, provided by [`scikit-image`](https://pypi.org/project/scikit-image/).

# %%
import skimage.data
import skimage.transform
import tensorflow as tf

grid_shape = [256, 256]
image = tf.convert_to_tensor(skimage.transform.resize(
    skimage.data.shepp_logan_phantom(), grid_shape).astype('float32'), dtype=tf.complex64)

print("image: \n - shape: {}\n - dtype: {}".format(image.shape, image.dtype))

# %% [markdown]
# Let us also create a k-space trajectory. In this example we will create a radial trajectory using the [`tensorflow-mri`](https://pypi.org/project/tensorflow-mri/) package, but you can create your own trajectories.

# %%
import tensorflow_mri as tfmr

points = tfmr.radial_trajectory(base_resolution=256, views=233)
points = tf.reshape(points, [-1, 2])

print("points: \n - shape: {}\n - dtype: {}\n - range: [{}, {}]".format(
    points.shape, points.dtype, tf.math.reduce_min(points), tf.math.reduce_max(points)))

# %% [markdown]
# Note that the trajectory should have shape `[..., M, N]`, where `M` is the number of points and `N` is the number of dimensions. Any additional outer dimensions will be treated as batch dimensions. Note that batch dimensions for `image` and `traj` will be automatically broadcasted.
# 
# In addition, note that spatial frequencies should be provided in radians/voxel, ie, in the range `[-pi, pi]`.
# %% [markdown]
# Finally, we'll also need density compensation weights for our set of nonuniform points. These are necessary in the adjoint transform, to compensate for the fact that the sampling density in a radial trajectory is not uniform. Here we use the `tensorflow-mri` package to calculate these weights.

# %%
weights = tfmr.radial_density(base_resolution=256, views=233)
weights = tf.reshape(weights, [-1])

print("weights: \n - shape: {}\n - dtype: {}".format(weights.shape, weights.dtype))

# %% [markdown]
# ## Forward transform (image to k-space)
# %% [markdown]
# Next, let's calculate the k-space coefficients for the given image and trajectory points (image to k-space transform).

# %%
import tensorflow_nufft as tfft

kspace = tfft.nufft(image, points, transform_type='type_2', j_sign='negative')

print("kspace: \n - shape: {}\n - dtype: {}".format(kspace.shape, kspace.dtype))

# %% [markdown]
# We are using a `'type_2'` transform (uniform to nonuniform) and a `'negative'` sign for the imaginary unit (signal domain to frequency domain). These are the default values for `transform_type` and `j_sign`, so it was not necessary to provide them in this case.
# %% [markdown]
# ## Adjoint transform (k-space to image)

# %%
# Apply density compensation.
comp_kspace = kspace * tf.cast(weights, tf.complex64)

recon = tfft.nufft(comp_kspace, points, transform_type='type_1', j_sign='positive', grid_shape=grid_shape)

print("recon: \n - shape: {}\n - dtype: {}".format(recon.shape, recon.dtype))


# %%
import matplotlib.pyplot as plt

plt.imshow(tf.abs(image))
plt.show()

plt.imshow(tf.abs(recon))
plt.show()


