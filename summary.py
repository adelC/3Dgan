import tensorflow as tf
import numpy as np

from utils import image_grid

def create_small_validation_summary(disc_loss, gen_loss, gp_loss, gen_sample, real_image_input):
    """Create a summary op for small summaries that are computed on the validation dataset.
    Parameters:
        disc_loss: discriminator loss
        gen_loss: generator loss
        gp_loss: gradient penalty loss
        gen_sample: (batch of) generated samples
        real_image_input: input tensor containing a batch of real images
    Returns:
        summary_small_validation: a single op that will update all small summaries
    """

    summary_small_validation = []

    with tf.name_scope('Loss/'):
        summary_small_validation.append(tf.summary.scalar('d_loss_val', disc_loss))
        summary_small_validation.append(tf.summary.scalar('g_loss_val', gen_loss))
        summary_small_validation.append(tf.summary.scalar('gp_val', tf.reduce_mean(gp_loss)))

    with tf.name_scope('Image_properties/'):
        summary_small_validation.append(tf.summary.scalar('image_min_real_val', tf.math.reduce_min(real_image_input[0])))
        summary_small_validation.append(tf.summary.scalar('image_max_real_val', tf.math.reduce_max(real_image_input[0])))

    summary_small_validation = tf.summary.merge(summary_small_validation)

    return summary_small_validation

def create_small_summary(disc_loss, gen_loss, gp_loss, g_gradients, g_variables, d_gradients, d_variables, max_g_norm, max_d_norm, gen_sample, real_image_input, energy_input, ang_input,  alpha, g_lr, d_lr):
    """Creates a summary op for small summaries, i.e. the ones that don't consume much disk space. These can be made frequently.
    Parameters:
        disc_loss: discriminator loss
        gen_loss: generator loss
        gp_loss: gradient penalty loss
        g_gradients: gradients for the generator
        g_variables: variable names corresponding to the g_gradients
        d_gradients: gradients for the discriminator
        d_variables: variable names corresponding to the d_gradients
        max_g_norm: the maximum norm of the generator gradients
        max_d_norm: the maximum norm of the discriminator gradients
        gen_sample: (batch of) generated samples
        real_image_input: input tensor containing a batch of real images
        energy_input:
        ang_input: 
        alpha: mixing factor alpha
        g_lr: generator learning rate
        d_lr: discriminator learning rate
    Returns:
        summary_small: a single op that will update all small summaries
    """

    summary_small = []

    with tf.name_scope('loss/'):

        summary_small.append(tf.summary.scalar('d_loss', disc_loss))
        summary_small.append(tf.summary.scalar('g_loss', gen_loss))
        summary_small.append(tf.summary.scalar('gp', tf.reduce_mean(gp_loss)))

        for g in zip(g_gradients, g_variables):
            summary_small.append(tf.summary.histogram(f'grad_{g[1].name}', g[0]))
        for g in zip(d_gradients, d_variables):
            summary_small.append(tf.summary.histogram(f'grad_{g[1].name}', g[0]))

        summary_small.append(tf.summary.scalar('max_g_grad_norm', max_g_norm))
        summary_small.append(tf.summary.scalar('max_d_grad_norm', max_d_norm))

            with tf.name_scope('Image_properties/'):
        summary_small.append(tf.summary.scalar('fake_image_max', tf.math.reduce_max(gen_sample)))	        summary_small.append(tf.summary.scalar('image_min_fake', tf.math.reduce_min(gen_sample)))
        summary_small.append(tf.summary.scalar('image_max_fake', tf.math.reduce_max(gen_sample)))

        summary_small.append(tf.summary.scalar('image_min_real', tf.math.reduce_min(real_image_input[0])))
        summary_small.append(tf.summary.scalar('image_max_real', tf.math.reduce_max(real_image_input[0])))
        
            with tf.name_scope('Training_properties/'):
        summary_small.append(tf.summary.scalar('alpha', alpha))

        summary_small.append(tf.summary.scalar('g_lr', g_lr))
        summary_small.append(tf.summary.scalar('d_lr', d_lr))

        summary_small = tf.summary.merge(summary_small)
            
        return summary_small


def create_large_summary(real_image_input, gen_sample):
    """Creates a summary op for large summaries, i.e. the ones that don't consume relatively large amounts of disc space. Should not be made too frequently.
    Parameters:
        real_image_input: input tensor containing a batch of real images
        gen_sample: (batch of) generated samples
    Returns:
        summary_large: a single op that will update all large summaries
    """

    summary_large = []

    with tf.name_scope('summaries'):
        # Spread out 3D image as 2D grid, slicing in the z-dimension
        real_image_grid = tf.transpose(real_image_input[0], (1, 2, 3, 0))
        shape = real_image_grid.get_shape().as_list()
        print(f'real_image_grid shape: {shape}')
        grid_cols = int(2 ** np.floor(np.log(np.sqrt(shape[0])) / np.log(2)))
        # If the image z-dimension isn't divisible by grid_rows, we need to pad
        if (shape[0] % grid_cols) != 0:
            # Initialize pad_list for numpy padding
            pad_list = [[0,0] for i in range(0, len(shape))]
            # Compute number of slices we need to add to get to the next multiple of shape[0]
            pad_nslices = grid_cols - (shape[0] % grid_cols)
            pad_list[0] = [0, pad_nslices]
            real_image_grid = tf.pad(real_image_grid, tf.constant(pad_list), "CONSTANT", constant_values=0)
            # Recompute shape, so that the number of grid_rows is adapted to that
            shape = real_image_grid.get_shape().as_list()
        grid_rows = int(np.ceil(shape[0] / grid_cols))
        grid_shape = [grid_rows, grid_cols]
        real_image_grid = image_grid(real_image_grid, grid_shape, image_shape=shape[1:3],
                                        num_channels=shape[-1])

        fake_image_grid = tf.transpose(gen_sample[0], (1, 2, 3, 0))
        # Use the same padding for the fake_image_grid
        if (fake_image_grid.get_shape().as_list()[0] % grid_cols) != 0:
            fake_image_grid = tf.pad(fake_image_grid, tf.constant(pad_list), "CONSTANT", constant_values=0)
        fake_image_grid = image_grid(fake_image_grid, grid_shape, image_shape=shape[1:3],
                                        num_channels=shape[-1])

        fake_image_grid = tf.clip_by_value(fake_image_grid, -1, 2)

        summary_large.append(tf.summary.image('real_image', real_image_grid))
        summary_large.append(tf.summary.image('fake_image', fake_image_grid))

        summary_large = tf.summary.merge(summary_large)

        return summary_large
