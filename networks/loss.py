import tensorflow as tf
import numpy as np
from networks.pgan.loss_utils import bce, mae, mape  # loss functions used for loss_fn='anglegan'
from networks.pgan.loss_utils import ecal_sum, ecal_angle # physics functions used for training the discriminator (should be in conditional lambda layer)


def forward_generator(generator,
                      discriminator,
                      real_image_input,
                      latent_dim,
                      alpha,
                      phase,
                      num_phases,
                      base_dim,
                      base_shape,
                      activation,
                      leakiness,
                      network_size,
                      loss_fn,
                      loss_weights,     #anglepgan#emmac
                      energy_input,              #anglepgan#emmac
                      ang_input,              #anglepgan#emmac   
                      noise_stddev,     #TODO
                      is_reuse=False
                      ):
    #anglepgan#emmac START
    z_batch_size = tf.shape(real_image_input)[0]                 # this value should be an integer

    #Adel is passing energy_input and ang as the correct batch sized numpy arrays
    energy_input_tensor = tf.reshape(energy_input, [z_batch_size,1])   # need z_batch_size x 1
    ang_input_tensor = tf.reshape(ang_input, [z_batch_size,1])   # need z_batch_size x 1

    ### OPTIONS FOR BUILDING THE LATENT SPACE ###
    latent_design = 'multiply by both energy and angle'
    if latent_design == 'concatenate energy and angle':  #254 random + energy + angle
      z = tf.random.normal(shape=[z_batch_size, latent_dim-2])
      z = tf.concat([z, energy_input_tensor, ang_input_tensor], 1)    # shape = (z_batch_size, 256)
    elif latent_design ==  'concatenate angle and multiply energy':   #try concatenating angle and then multiplying by energy?
      z = tf.random.normal(shape=[z_batch_size, latent_dim-1])
      z = tf.concat([z, ang_input_tensor], 1) 
      z = tf.math.multiply(energy_input_tensor, z)   #try multiplying the latent space by the energy...tf.math.multiply should take care of broadcasting
    elif latent_design == 'multiply by both energy and angle':
      z = tf.random.normal(shape=[z_batch_size, latent_dim])
      z = tf.math.multiply(energy_input_tensor, z)   #try multiplying the latent space by the energy...tf.math.multiply should take care of broadcasting
      z = tf.math.multiply(ang_input_tensor, z)   #try multiplying the latent space by the angle...tf.math.multiply should take care of broadcasting
    else:    #default is z=256 random
      z = tf.random.normal(shape=[z_batch_size, latent_dim])
      
    #anglepgan#emmac END
    
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation=activation,
                           param=leakiness, size=network_size, is_reuse=is_reuse)
    
    #TODO - understand this and find noise value for cern data
    # Add instance noise to make training more stable. See e.g. https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    #real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * noise_stddev
    gen_sample_noisy = gen_sample #+ tf.random.normal(shape=tf.shape(gen_sample)) * noise_stddev

    # Generator training. #anglepgan#emmac
    disc_fake_g, fake_ecal, fake_ang = discriminator(gen_sample, alpha, phase, num_phases, base_shape, base_dim, latent_dim,
                                activation=activation, param=leakiness, size=network_size, is_reuse=is_reuse)
    if loss_fn == 'wgan':
        gen_loss = -tf.reduce_mean(disc_fake_g)

    elif loss_fn == 'logistic':
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))
  
    #anglepgan#emmac#ach
    elif loss_fn == 'anglegan':
        gen_loss = tf.reduce_mean(-disc_fake_g) #TODO
      
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return gen_sample, gen_loss


def forward_discriminator(generator,
                          discriminator,
                          real_image_input,
                          latent_dim,
                          alpha,
                          phase,
                          num_phases,
                          base_dim,
                          base_shape,
                          activation,
                          leakiness,
                          network_size,
                          loss_fn,
                          gp_weight,
                          loss_weights,   #anglepgan#emmac
                          energy_input,            #anglepgan#emmac
                          ang_input,            #anglepgan#emmac
                          noise_stddev,   #TODO
                          is_reuse=False,
                          ):
    
    #anglepgan#emmac START
    z_batch_size = tf.shape(real_image_input)[0]                 # this value should be an integer

    #Adel is passing e_p and ang as the correct batch sized numpy arrays
    enery_input_tensor = tf.reshape(enery_input, [z_batch_size,1])   # need z_batch_size x 1
    ang_input_tensor = tf.reshape(ang_input, [z_batch_size,1])   # need z_batch_size x 1

    z = tf.random.normal(shape=[z_batch_size, latent_dim-2])
    z = tf.concat([z, enery_input_tensor, ang_input_tensor], 1)    # shape = (z_batch_size, 256)
    #anglepgan#emmac END    
    
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation=activation,
                           param=leakiness, size=network_size, is_reuse=is_reuse)
    
    #TODO - understand this and find noise value for cern data
    # Add instance noise to make training more stable. See e.g. https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    #real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * noise_stddev
    gen_sample_noisy = gen_sample #+ tf.random.normal(shape=tf.shape(gen_sample)) * noise_stddev

    # Discriminator Training
    disc_fake_d = discriminator(tf.stop_gradient(gen_sample_noisy), alpha, phase, num_phases,
                                base_shape, base_dim, latent_dim, activation=activation, param=leakiness,
                                size=network_size, )
    disc_real = discriminator(real_image_input, alpha, phase, num_phases,
                              base_shape, base_dim, latent_dim, activation=activation, param=leakiness,
                              is_reuse=True, size=network_size, )

    gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = gamma * real_image_input + (1 - gamma) * tf.stop_gradient(gen_sample_noisy)
    
    #anglepgan#ach
    disc_fake_d2, fake_ecal2, fake_ang2 = discriminator(interpolates, alpha, phase,
                                           num_phases, base_shape, base_dim, latent_dim,
                                           is_reuse=True, activation=activation,
                                           param=leakiness, size=network_size, )
    gradients = tf.gradients(disc_fake_d2, [interpolates])[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3, 4)))
    
    #anglepgan#ach
    print(f"ANGLEPGAN DEBUG ### : disc_fake_d={disc_fake_d}, disc_fake_d.shape={disc_fake_d.shape}")
    print(f"ANGLEPGAN DEBUG ### : gradients={gradients}, gradients.shape={gradients.shape}")
    print(f"ANGLEPGAN DEBUG ### : slopes={slopes}, slopes.shape={slopes.shape}")

    if loss_fn == 'wgan':
        # has the real/fake activation layer built in, as a critic that rates. Allows you to move far away and still get a good gradient
        gradient_penalty = (slopes - 1) ** 2
        gp_loss = gp_weight * gradient_penalty
        disc_loss = disc_fake_d - disc_real
        drift_loss = 1e-3 * disc_real ** 2
        disc_loss = tf.reduce_mean(disc_loss + gp_loss + drift_loss)

    elif loss_fn == 'logistic':
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        gp_loss = gp_weight * gradient_penalty
        disc_loss = tf.reduce_mean(tf.nn.softplus(disc_fake_d)) + tf.reduce_mean(
            tf.nn.softplus(-disc_real)) #TODO check
        disc_loss += gp_loss

    #anglepgan#emmac#ach START
    elif loss_fn == 'anglegan':   #TODO NEEDS TO BE TESTED + DEBUGGED!!
        fake_loss = bce(disc_real, disc_fake_d)  
        # need to use ecal_angle() in conditional lambda layer (in d) to find ang_output/ang_target
        ang_loss = mae(real_ang, fake_ang)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ANGS?!!
        # need to use ecal_sum() in conditional lambda layer (in d) to find ecal_output/ecal_target
        ecal_loss = mape(real_ecal, fake_ecal)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ECALS?!!

        losses = np.array([fake_loss, ang_loss, ecal_loss])   # calculate the losses and store in an array
        loss_weights = loss_weights.numpy() # make sure pgan weight vector is a np.array
        disc_loss = np.dot(loss_weights, losses)  # weight and sum the losses

        gp_loss = gp_weight * gradient_penalty   
        disc_loss += gp_loss    
        #anglepgan#emmac#ach
        
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return disc_loss, gp_loss


def forward_simultaneous(generator,
                         discriminator,
                         real_image_input,
                         latent_dim,
                         alpha,
                         phase,
                         num_phases,
                         base_dim,
                         base_shape,
                         activation,
                         leakiness,
                         network_size,
                         loss_fn,
                         gp_weight,
                         loss_weights,     #anglepgan#emmac
                         energy_input,     #anglepgan#emmac
                         ang_input,     #anglepgan#emmac
                         noise_stddev,
                         conditioning=None
                         ):
    
    #anglepgan#emmac#ach START
    z_batch_size = tf.shape(real_image_input)[0]                 # this value should be an integer

    #Adel is passing e_p and ang as the correct batch sized numpy arrays
    energy_input_tensor = tf.reshape(energy_input, [z_batch_size,1])   # need z_batch_size x 1
    ang_input_tensor = tf.reshape(ang_input, [z_batch_size,1])   # need z_batch_size x 1

    z = tf.random.normal(shape=[z_batch_size, latent_dim-2])
    z = tf.concat([z, energy_input_tensor, ang_input_tensor], 1)    # shape = (z_batch_size, 256)    
    
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation=activation,
                           param=leakiness, size=network_size, conditioning=conditioning)
    #anglepgan#emmac#ach END
    
    #TODO - understand this and find noise value for cern data
    # Add instance noise to make training more stable. See e.g. https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    #real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * noise_stddev
    gen_sample_noisy = gen_sample #+ tf.random.normal(shape=tf.shape(gen_sample)) * noise_stddev

    # Discriminator Training   #TODO - do we need to feed through the ng and ecal and loss weights here???
    disc_fake_d, fake_ecal, fake_ang = discriminator(tf.stop_gradient(gen_sample_noisy), alpha, phase, num_phases,
                                base_shape, base_dim, latent_dim, activation=activation, param=leakiness,
                                size=network_size, conditioning=conditioning)
    disc_real, real_ecal, real_ang  = discriminator(real_image_input, alpha, phase, num_phases,
                              base_shape, base_dim, latent_dim, activation=activation, param=leakiness,
                              is_reuse=True, size=network_size, conditioning=conditioning)
    
    gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = gamma * real_image_input + (1 - gamma) * tf.stop_gradient(gen_sample_noisy)

    gradients = tf.gradients(discriminator(interpolates, alpha, phase,
                                           num_phases, base_shape, base_dim, latent_dim,
                                           is_reuse=True, activation=activation,
                                           param=leakiness, size=network_size, conditioning=conditioning), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3)))

     # Generator training. #anglegan#emmac#ach
    disc_fake_g, fake_ecal_g, fake_ang_g = discriminator(gen_sample, alpha, phase, num_phases, base_shape, base_dim, latent_dim,
                                activation=activation, param=leakiness, size=network_size, is_reuse=True, conditioning=conditioning)

    if loss_fn == 'wgan':
        gradient_penalty = (slopes - 1) ** 2
        gp_loss = gp_weight * gradient_penalty
        disc_loss = disc_fake_d - disc_real
        drift_loss = 1e-3 * disc_real ** 2
        disc_loss = tf.reduce_mean(disc_loss + gp_loss + drift_loss)
        gen_loss = -tf.reduce_mean(disc_fake_g)

    elif loss_fn == 'logistic':
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        gp_loss = gp_weight * gradient_penalty
        disc_loss = tf.reduce_mean(tf.nn.softplus(disc_fake_d)) + tf.reduce_mean(
            tf.nn.softplus(-disc_real))
        disc_loss += gp_loss
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))
   
  #################################################anglepgan#ach#emmac START
     
    elif loss_fn == 'anglegan':   # NEEDS TO BE TESTED + DEBUGGED!!
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        fake_loss = bce(disc_real, disc_fake_d)     # NOT SURE!! CHECK!
        # need to use ecal_angle() in conditional lambda layer (in d) to find ang_output/ang_target
        ang_loss = mae(real_ang, fake_ang)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ANGS?!!
        # need to use ecal_sum() in conditional lambda layer (in d) to find ecal_output/ecal_target
        ecal_loss = mape(real_ecal, fake_ecal)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ECALS?!!

        print(f"ANGLEPGAN DEBUG ### : fake_loss={fake_loss}")
        print(f"ANGLEPGAN DEBUG ### : ang_loss={ang_loss}")
        print(f"ANGLEPGAN DEBUG ### : ecal_loss={ecal_loss}")
        print(f"ANGLEPGAN DEBUG ### : loss_weights={loss_weights}")

        losses = np.array([fake_loss, ang_loss, ecal_loss])   # calculate the losses and store in an array
        loss_weights = np.array(loss_weights) # make sure pgan weight vector is a np.array
        disc_loss = np.dot(loss_weights, losses)  # weight and sum the losses

        gp_loss = gp_weight * gradient_penalty   #HOW TO IMPLEMENT THIS???
        disc_loss += gp_loss    #HOW TO IMPLEMENT THIS???
        disc_loss = disc_loss[0]
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))

    elif loss_fn =='anglegan2':
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        gp_loss = gp_weight * gradient_penalty
        disc_loss = tf.reduce_mean(tf.nn.softplus(disc_fake_d)) + tf.reduce_mean(tf.nn.softplus(-disc_real))
        disc_loss += gp_loss

        ang_loss = mae(real_ang, fake_ang)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ANGS?!!
        ecal_loss = mape(real_ecal, fake_ecal)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ECALS?!!

        ## ANGLEPGAN ToDO ; ecal_sum retunr (batch,1) and discriminator return (batch,) ?? same thing for angle
        ## that's why we do expand_dim here
        ang_loss = tf.expand_dims(ang_loss, 1)
        ecal_loss = tf.expand_dims(ecal_loss, 1)

        disc_loss *= loss_weights[0]
        ang_loss *= loss_weights[1]
        ecal_loss *= loss_weights[2]

        disc_loss = tf.reduce_mean(disc_loss) + tf.reduce_mean(ang_loss) + tf.reduce_mean(ecal_loss)
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))

        print(f"ANGLEPGAN DEBUG ## : disc_loss={disc_loss}")
        print(f"ANGLEPGAN DEBUG ## : ang_loss={ang_loss}")
        print(f"ANGLEPGAN DEBUG ## : ecal_loss={ecal_loss}")
        print(f"ANGLEPGAN DEBUG ## : gen_loss={gen_loss}")
        #################################################anglepgan#ach#emmac END

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return gen_loss, disc_loss, gp_loss, gen_sample, ang_loss, ecal_loss
