# # pylint: disable=import-error
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import time
import random
import logging
import copy

from utils import count_parameters, parse_tuple, MPMap, log0
from utils import get_compute_metrics_dict, get_logdir, get_verbosity, get_filewriter, get_base_shape, get_num_phases, get_num_channels    #caspar TODO check get_num_phases, get_base_shape
from utils import get_num_metric_samples, scale_lr, get_xy_dim, get_numpy_dataset, get_current_input_shape, restore_variables, print_summary_to_stdout
import dataset as data
from optuna_suggestions import optuna_override_undefined
import os
import importlib
from networks.loss import forward_simultaneous, forward_generator, forward_discriminator
from dataset import NumpyPathDataset
from metrics.save_metrics import save_metrics

import networks as nw
import optimization as opt
import summary

def optuna_objective(trial, args, config):
    # We support several types of runs (see main.py)
    run_from_best_trial = (args.optuna_use_best_trial and (args.optuna_storage is not None))
    hyperparam_opt_inter_trial = args.optuna_distributed and not run_from_best_trial
    hyperparam_opt_intra_trial = (args.optuna_storage is not None) and (
                args.optuna_study_name is not None) and not hyperparam_opt_inter_trial
    normal_run = (not run_from_best_trial) and (not hyperparam_opt_inter_trial) and (not hyperparam_opt_intra_trial)

    # Store the last fid so that it can be returned to optuna
    last_fid = None

    # Preserve original arguments by creating a deepcopy of args, and call optuna_override_undefined on that.
    # Otherwise, when running multiple trials, the second trial will always get the arguments from the first trial, rather than calling new suggest_*'s
    args_copy = copy.deepcopy(args)

    # Override args.* that are undefined by optuna's suggest_* calls.
    # For now, this is limited to overriding learning rate, batch size, and learning rate schedules, but may be expanded in the future (see optuna_suggestions.py)
    # Note: this means that when restoring from an optuna FrozenTrial, command line parameters take precedence!
    args = optuna_override_undefined(args_copy, trial)

    # If tuning hyperparameters with intra-trial parallelism, send the trial and args object so that the other workers can call optuna_objective with those as arguments
    if hyperparam_opt_intra_trial and hvd.rank() == 0:
        MPI.COMM_WORLD.bcast(trial, root=0)
        print(f'Worker {hvd.rank()} sending args: {args}')
        MPI.COMM_WORLD.bcast(args, root=0)

    # Importing modules by name for the generator and discriminator
    discriminator = importlib.import_module(f'networks.{args.architecture}.discriminator').discriminator
    generator = importlib.import_module(f'networks.{args.architecture}.generator').generator

    # Set verbosity:
    verbose = get_verbosity(args.horovod, args.optuna_distributed)
    if not verbose:
        tf.get_logger().setLevel(logging.ERROR)  # Only errors if rank != 0

    # set world size
    if args.horovod:
        global_size = hvd.size()
    else:
        global_size = 1

    # Get logging directory based on the args. If args.logdir is not set, a logdir is created
    logdir = get_logdir(args)
    # Returns a tf.FileWriter, but only for rank 0 if the training uses multiple MPI ranks
    writer = get_filewriter(logdir, verbose)

    # Allow GPU memory growth
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Number of filters at the base (= 1st convolutional layer of the generator) of the progressive network
    # In subsequent phases, the number of filters will go down as the resolution goes up.
    base_dim = args.first_conv_nfilters

    if verbose:
        #print(f"Start resolution: {start_resolution}")   #anglepgan#ach
        #print(f"Final resolution: {final_resolution}")   #anglepgan#ach
        #print(f"Deduced number of phases: {num_phases}") #anglepgan#ach
        print(f"Deduced number of phases: {get_num_phases(args.start_shape, args.final_shape)}")
        print(f"base_dim: {base_dim}")

    var_list = list()
    global_step = 0
    
    #anglepgan#ach
    energy_path = os.path.join(args.dataset_path, f'en/')
    ang_path = os.path.join(args.dataset_path, 'ang/')
    ecal_path = os.path.join(args.dataset_path, 'ecal/')
    
    
    #anglepgan #ach : reading en, ang and ecal once, so out of the loop, not phase dependent 
    npy_energy = NumpyPathDataset(energy_path, args.scratch_path, copy_files=hvd.local_rank == 0, is_correct_phase=True)
    #anglepgan#ach
    npy_ang = NumpyPathDataset(ang_path, args.scratch_path, copy_files=hvd.local_rank == 0, is_correct_phase=True)
        
    #anglepgan#ach
    npy_ecal = NumpyPathDataset(ecal_path, args.scratch_path, copy_files=hvd.local_rank == 0, is_correct_phase=True)

    # Loop over the different phases (resolutions) of training of a progressive architecture
    for phase in range(1, get_num_phases(args.start_shape, args.final_shape) + 1):

        tf.reset_default_graph()
        
        # Random seeds need to be reinitialized after a reset_default_graph (at least for TF, but I guess resetting all is good)
        if args.horovod:
            np.random.seed(args.seed + hvd.rank())
            tf.random.set_random_seed(args.seed + hvd.rank())
            random.seed(args.seed + hvd.rank())
        else:
            np.random.seed(args.seed)
            tf.random.set_random_seed(args.seed)
            random.seed(args.seed)

        # ------------------------------------------------------------------------------------------#
        # DATASET

        # Get NumpyPathDataset object for current phase. It's an iterable object that returns the path to samples in the dataset
        # caspar method (#TODO check) - npy_data = get_numpy_dataset(phase, args.starting_phase, args.start_shape, args.dataset_path, args.scratch_path, verbose)
        
        #AnglePGAN #ACH --- Commenting bloc code / Begin
        #size = start_resolution * (2 ** (phase - 1))
       
        #anglepgan#ach
        #data_path = os.path.join(args.dataset_path, f'{size}x{size}/')
        
        #anglepgan#ach
        #npy_data = NumpyPathDataset(data_path, args.scratch_path, copy_files=local_rank == 0,
        #                            is_correct_phase=phase >= args.starting_phase)
        
        #AnglePGAN #ACH: Begin - better adapting  get_numpy_dataset if needed rather then using NumpyPathDataset
        npy_data = get_numpy_dataset(phase, args.starting_phase, args.start_shape, args.dataset_path, args.scratch_path, verbose)
        
        #anglepgan#ach
        #npy_en = NumpyPathDataset(en_path, args.scratch_path, copy_files=local_rank == 0,
        #                            is_correct_phase=phase >= args.starting_phase)
        #anglepgan#ach
        #npy_ang = NumpyPathDataset(ang_path, args.scratch_path, copy_files=local_rank == 0,
        #                            is_correct_phase=phase >= args.starting_phase)
        
        #anglepgan#ach
        #npy_ecal = NumpyPathDataset(ecal_path, args.scratch_path, copy_files=local_rank == 0,
        #                            is_correct_phase=phase >= args.starting_phase)
        
        #anglepgan#ach
        #if verbose:
        #    print(f'Phase {phase}: reading data from dir {data_path}')
        #ach erro : npy_data doubled - to be removed
        #npy_data = NumpyPathDataset(data_path, args.scratch_path, copy_files=local_rank == 0,
        #                            is_correct_phase=phase >= args.starting_phase)
        
        #AnglePGAN #ACH --- Commenting bloc code / End
          
        #AnglePGAN #ACH - better adapting  get_numpy_dataset if needed rather then using NumpyPathDataset
        #caspar #surfgan #TODO choose over line 116?
        # Get NumpyPathDataset object for current phase. It's an iterable object that returns the path to samples in the dataset
        
        
        #CASPAR #TODO: we should probably split the npy_data in a train and validation set. The validation set can then be passed to save_metrics to compute the metrics on.
        # Note: the split below preserves the ordering of npy_data. Thus, similar filenames tend to either end up all in the training or validation set.
        # That may or may not make much sense, depending on whether there is correlation between your samples!
        # For the medical CT scans there was: some scans are from the same patient, and usually have consequtive numbering.
        # By splitting this way, we avoid as much as possible that correlated scans end up in both training and validation sets.
        npy_data_train, npy_data_testval = npy_data.split_by_fraction(1 - (args.validation_fraction + args.test_fraction))
        npy_data_validation, npy_data_test = npy_data_testval.split_by_fraction(args.validation_fraction / (args.validation_fraction + args.test_fraction))
        
        #ach #anglepgan - Begin
        npy_energy_train, npy_energy_testval = npy_energy.split_by_fraction(1 - (args.validation_fraction + args.test_fraction))
        npy_energy_validation, npy_energy_test = npy_energy_testval.split_by_fraction(args.validation_fraction / (args.validation_fraction + args.test_fraction))
        
        npy_ang_train, npy_ang_testval = npy_ang.split_by_fraction(1 - (args.validation_fraction + args.test_fraction))
        npy_ang_validation, npy_ang_test = npy_ang_testval.split_by_fraction(args.validation_fraction / (args.validation_fraction + args.test_fraction))
        
        npy_ecal_train, npy_ecal_testval = npy_ecal.split_by_fraction(1 - (args.validation_fraction + args.test_fraction))
        npy_ecal_validation, npy_ecal_test = npy_ecal_testval.split_by_fraction(args.validation_fraction / (args.validation_fraction + args.test_fraction))
        #ach #anglepgan - End
        
        if verbose:
            print(f"Split dataset of {len(npy_data)} samples: train {len(npy_data_train)}, validation {len(npy_data_validation)}, test {len(npy_data_test)}")

        # Get DataLoader
        batch_size = max(1, args.base_batch_size // (2 ** (phase - 1)))

        if phase >= args.starting_phase:
            # assert batch_size * global_size <= args.max_global_batch_size
            if verbose:
                print(f"Using local batch size of {batch_size} and global batch size of {batch_size * global_size}")

        # Num_metric_samples is the amount of samples the metric is calculated on.
        # If it is not set explicitely, we use the same as the global batch size, but never less than 2 per worker (1 per worker potentially makes some metrics crash)
        num_metric_samples = get_num_metric_samples(args.num_metric_samples, batch_size, global_size)

        # Create input tensor
        real_image_input = tf.placeholder(shape=get_current_input_shape(phase, batch_size, args.start_shape), dtype=tf.float32) #TODO check shape function
        
        #anglepgan #ach - creating imput tensor for en ang ang (dataset used to condition the GAN training) #ToDo
        #energy_input = tf.placeholder(shape=get_current_input_shape(phase, batch_size, args.start_shape), dtype=tf.float32) #TODO check shape function
        #angle_input = tf.placeholder(shape=get_current_input_shape(phase, batch_size, args.start_shape), dtype=tf.float32) #TODO check shape function
        
        #anglepgan #ach
        energy_shape = [batch_size, 1]
        ang_shape = [batch_size, 1]
        energy_input = tf.placeholder(shape=energy_shape, dtype=tf.float32, name="energy_input")
        ang_input = tf.placeholder(shape=ang_shape, dtype=tf.float32, name="ang_input")
        #anglepgan #ach
        
        real_label = None
        if real_label is not None:
            real_label = tf.one_hot(real_label, depth=args.num_labels)
            
        # ------------------------------------------------------------------------------------------#
        # OPTIMIZERS

        # Scale learning rate to compensate for data parallel training, if a scaling strategy is specified
        g_lr, d_lr = scale_lr(args.g_lr, args.d_lr, args.g_scaling, args.d_scaling, args.horovod)   #TODO check this function, missing lines 191-211 in adel/pgan/main

        d_lr = tf.Variable(d_lr, name='d_lr', dtype=tf.float32)
        g_lr = tf.Variable(g_lr, name='g_lr', dtype=tf.float32)
        
        #TODO - make this customizable through parameter
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=args.beta1, beta2=args.beta2)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=args.beta1, beta2=args.beta2)
        # optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=g_lr)
        # optimizer_disc = tf.train.RMSPropOptimizer(learning_rate=d_lr)
        # optimizer_gen = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        # optimizer_disc = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        # optimizer_gen = RAdamOptimizer(learning_rate=g_lr, beta1=args.beta1, beta2=args.beta2)
        # optimizer_disc = RAdamOptimizer(learning_rate=d_lr, beta1=args.beta1, beta2=args.beta2)

        intra_phase_step = tf.Variable(0, name='step', dtype=tf.int32)
        update_intra_phase_step = intra_phase_step.assign_add(batch_size * global_size)

        # Turn arguments into constant Tensors
        g_lr_max = tf.constant(args.g_lr, tf.float32)
        d_lr_max = tf.constant(args.d_lr, tf.float32)
        #TODO - #anglepgan#ach has rise_niter and decay_niter that is taken out (lines 231-234) I think for optuna?
        steps_per_phase = tf.constant(args.mixing_nimg + args.stabilizing_nimg)

        update_g_lr = opt.lr_update(lr = g_lr, intra_phase_step = intra_phase_step, 
                                     steps_per_phase = steps_per_phase, lr_max = g_lr_max,
                                     lr_increase = args.g_lr_increase, lr_decrease = args.g_lr_decrease,
                                     lr_rise_niter = args.g_lr_rise_niter, lr_decay_niter = args.g_lr_decay_niter
                                    )
        update_d_lr = opt.lr_update(lr = d_lr, intra_phase_step = intra_phase_step, 
                                     steps_per_phase = steps_per_phase, lr_max = d_lr_max,
                                     lr_increase = args.d_lr_increase, lr_decrease = args.d_lr_decrease,
                                     lr_rise_niter = args.d_lr_rise_niter, lr_decay_niter = args.d_lr_decay_niter
                                    )

        if args.horovod:
            if args.use_adasum:
                # optimizer_gen = hvd.DistributedOptimizer(optimizer_gen, op=hvd.Adasum)
                optimizer_gen = hvd.DistributedOptimizer(optimizer_gen)
                optimizer_disc = hvd.DistributedOptimizer(optimizer_disc, op=hvd.Adasum)
            else:
                optimizer_gen = hvd.DistributedOptimizer(optimizer_gen)
                optimizer_disc = hvd.DistributedOptimizer(optimizer_disc)

        # ------------------------------------------------------------------------------------------#
        # NETWORKS

        with tf.variable_scope('alpha'):
            alpha = tf.Variable(1, name='alpha', dtype=tf.float32)

        # Alpha ops
        init_alpha = alpha.assign(1)
        update_alpha = nw.ops.alpha_update(alpha, args.mixing_nimg, args.starting_alpha, batch_size, global_size)
        assign_starting_alpha = alpha.assign(args.starting_alpha)
        assign_zero = alpha.assign(0)
        
        #Caspar changes structure and uses new helper functions like get_num_phases #TODO check this runs smoothly
        # Performs a forward pass, computes gradients, clips them (if desired), and then applies them.
        # Supports simultaneous forward pass of generator and discriminator, or alternatingly (discriminator first)
        train_gen, train_disc, gen_loss, disc_loss, gp_loss, gen_sample, g_gradients, g_variables, d_gradients, d_variables, max_g_norm, max_d_norm = opt.optimize_step(
            optimizer_gen,
            optimizer_disc,
            generator,
            discriminator,
            real_image_input,
            energy_input,  #TODO #anglepgan #ach check placement
            ang_input, #TODO #anglepgan #ach check placement
            args.latent_dim,
            alpha,
            phase,
            get_num_phases(args.start_shape, args.final_shape),
            base_dim,
            get_base_shape(args.start_shape),
            args.activation,
            args.leakiness,
            args.network_size,
            args.loss_fn,
            args.loss_weights,
            args.gp_weight,
            args.optim_strategy,
            args.g_clipping,
            args.d_clipping,
            args.noise_stddev
        )

        if verbose:
            print(f"Generator parameters: {count_parameters('generator')}")
            print(f"Discriminator parameters:: {count_parameters('discriminator')}")

        # Create an exponential moving average of generator weights. We update this every step.
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        
        #TODO - check #Caspar's removal of main.py lines 300-388 
        
        ema = tf.train.ExponentialMovingAverage(decay=args.ema_beta)
        ema_op = ema.apply(gen_vars)
        # After training has completed, we copy the EMA weights to the generator using the following op (before saving the generator model).
        ema_update_weights = tf.group(
            [tf.assign(var, ema.average(var)) for var in gen_vars])

        # ------------------------------------------------------------------------------------------#
        # Summaries

        summary_small = summary.create_small_summary(
            disc_loss,
            gen_loss,
            gp_loss,
            g_gradients,
            g_variables,
            d_gradients,
            d_variables,
            max_g_norm,
            max_d_norm,
            gen_sample,
            real_image_input,
            energy_input,
            ang_input,
            alpha,
            g_lr,
            d_lr
        )
        #TODO check the new summary function has adel/pgan/main.py lines 403-468
        # This is only computed on the validation dataset
        summary_small_validation = summary.create_small_validation_summary(
            disc_loss,
            gen_loss,
            gp_loss,
            gen_sample,
            real_image_input,
        )
        summary_large = summary.create_large_summary(
            real_image_input,
            gen_sample
        )

        # ------------------------------------------------------------------------------------------#
        # Other ops

        init_op = tf.global_variables_initializer()
        #TODO check #Caspar's removal of assign_starting_alpha and assign_zero
        # Probably these alpha ops could be with the other ops above, but... it changes reproducibility of my runs. So for now, I'll leave them here.

        broadcast = hvd.broadcast_global_variables(0)

        with tf.Session(config=config) as sess:
            # if args.gpu:
            #     assert tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
            # sess.graph.finalize()
            sess.run(init_op)
            
            #Caspar cut this section down a lot #TODO check that it runs smoothly (adel/pgan/main l484-504)
            # Do variables need to be restored? (either from the previous phase, or from a previous run)
            if (phase > args.starting_phase) or (args.continue_path and phase == args.starting_phase):
                restore_variables(sess, phase, args.starting_phase, logdir, args.continue_path, var_list, verbose)
            else:
                if verbose:
                    print("Not restoring variables.")
                    writer.add_graph(sess.graph)

            # Store the variable list in this phase. This is the list that needs to be loaded in the next phase.
            # That way, only the newly added layers will have randomly initialized weights, while the other layers will get their weights set by the restore_variables function.
            var_list = gen_vars + disc_vars

            # If we haven't reached the starting phase, simply continue with the next loop iterations
            if phase < args.starting_phase:
                continue

            if phase == args.starting_phase:
                sess.run(assign_starting_alpha)
            else:
                sess.run(init_alpha)

            if verbose:
                print(f"Begin mixing epochs in phase {phase}")
            if args.horovod:
                if verbose:
                    print("Broadcasting initial global variables...")
                sess.run(broadcast)
                if verbose:
                    print("Broadcast completed")

            local_step = 0
            # take_first_snapshot = True

            if trial is not None:
                # Only for inter-trial parallelism: each worker has its own trial, so should report its own parameters. Otherwise, only rank 0 should report
                if hyperparam_opt_inter_trial:
                    print(f"Trial: {trial.number}, Worker: {hvd.rank()}, Parameters: {trial.params}")
                elif hvd.rank() == 0:
                    print(f"Trial: {trial.number}, Parameters: {trial.params}")

            # ------------------------------------------------------------------------------------------#
            # Training loop for mixing phase
            
            #Caspar added #TODO check it works
            # Do we start with a mixing phase? (normally we do, unless we resume e.g. from a point in the stabilization phase)
            if args.mixing_nimg > 0:
                mixing_bool = True

            while True:
                start = time.time()

                # Update learning rate
                d_lr_val = sess.run(update_d_lr)
                g_lr_val = sess.run(update_g_lr)

                if not mixing_bool:
                    assert alpha.eval() == 0

                if global_step % args.checkpoint_every_nsteps < (batch_size * global_size) and local_step > 0:
                    if args.horovod:
                        if verbose:
                            print("Broadcasting global variables for checkpointing...")
                        sess.run(broadcast)
                        if verbose:
                            print("Broadcast completed")
                    saver = tf.train.Saver(var_list)
                    if verbose:
                        print(f'Writing checkpoint file: model_{phase}_ckpt_{global_step}')
                        saver.save(sess, os.path.join(logdir, f'model_{phase}_ckpt_{global_step}'))

                # Get randomly selected batch
                batch = npy_data_train.batch(batch_size)
                
                #anglepgan #ach
                batch_energy = npy_energy.batch(batch_size)
                batch_ang = npy_ang.batch(batch_size)
                batch_ecal = npy_ecal.batch(batch_size)
            
                # Normalize data (but only if args.data_mean AND args.data_stddev are defined
                batch = data.normalize_numpy(batch, args.data_mean, args.data_stddev, verbose)

                #anglepgan#ach begin #TODO
                #batch_loc_en = np.random.randint(0, len(npy_en) - batch_size)
                #batch_paths_en = npy_en[batch_loc_en: batch_loc_en + batch_size]
                #batch_en = np.stack([np.load(path) for path in batch_paths_en])
                #batch_en = batch_en[:, np.newaxis, ...].astype(np.float32) / 1024 - 1

                #batch_loc_ang = np.random.randint(0, len(npy_ang) - batch_size)
                #batch_paths_ang = npy_ang[batch_loc_ang: batch_loc_ang + batch_size]
                #batch_ang = np.stack([np.load(path) for path in batch_paths_ang])
                #batch_ang = batch_ang[:, np.newaxis, ...].astype(np.float32) / 1024 - 1

                #batch_loc_ecal = np.random.randint(0, len(npy_ecal) - batch_size)
                #batch_paths_ecal = npy_ecal[batch_loc_ecal: batch_loc_ecal + batch_size]
                #batch_ecal = np.stack([np.load(path) for path in batch_paths_ecal])
                #batch_ecal = batch_ecal[:, np.newaxis, ...].astype(np.float32) / 1024 - 1
                #anglepgan#ach end

                #COMMENTED THIS OUT, #TODO CHECK - bc sofia said normalizing may mess with our data bc of the wide range of values
                # Normalize data (but only if args.data_mean AND args.data_stddev are defined)
                batch = data.normalize_numpy(batch, args.data_mean, args.data_stddev, verbose)

                # if args.horovod:
                #     print(f"Worker {hvd.rank()} got batch from {batch_loc} to {batch_loc + batch_size}")
                # else:
                #     print(f"got batch from {batch_loc} to {batch_loc + batch_size}")
                # print("Got a batch!")

                # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
                # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6789')

                # Measure speed as often as small_summaries, but one iteration later. This avoids timing the summaries themselves.
                speed_measurement_bool = ((local_step - 1) % args.summary_small_every_nsteps < batch_size)
                small_summary_bool = (local_step % args.summary_small_every_nsteps < batch_size)
                large_summary_bool = (local_step % args.summary_large_every_nsteps < batch_size)
                metrics_summary_bool = (local_step % args.metrics_every_nsteps < batch_size)

                # Run training step, including summaries
                if large_summary_bool:
                    _, _, summary_s, summary_l, d_loss, g_loss = sess.run(
                         [train_gen, train_disc, summary_small, summary_large,
                          disc_loss, gen_loss], feed_dict={real_image_input: batch, energy_input : batch_energy, ang_input : batch_ang}) #anglepgan #ach #ToDo
                elif small_summary_bool:
                    _, _, summary_s, d_loss, g_loss = sess.run(
                         [train_gen, train_disc, summary_small,
                          disc_loss, gen_loss], feed_dict={real_image_input: batch, energy_input : batch_energy, ang_input : batch_ang}) #anglepgan #ach #ToDo
                else:
                    _, _, d_loss, g_loss = sess.run(
                         [train_gen, train_disc, disc_loss, gen_loss],
                         feed_dict={real_image_input: batch, energy_input : batch_energy, ang_input : batch_ang})
               
                # Run validation loss
                if large_summary_bool or small_summary_bool:
                    batch_val = npy_data_validation.batch(batch_size)
                    
                    #anglepgan #ach #ToDo
                    batch_energy_val = npy_energy_validation.batch(batch_size)
                    batch_ang_val = npy_ang_validation.batch(batch_size)
                    batch_ecal_val = npy_ecal_validation.batch(batch_size)
                  
                    batch_val = data.normalize_numpy(batch_val, args.data_mean, args.data_stddev, verbose)
                    summary_s_val = sess.run(summary_small_validation, feed_dict={real_image_input: batch_val, energy_input : batch_energy_val, ang_input : batch_ang_val} ) #anglepgan #ach #ToDo
                  
                #print("Completed step")
                global_step += batch_size * global_size
                local_step += batch_size

                end = time.time()
                local_img_s = batch_size / (end - start)
                img_s = global_size * local_img_s

                if mixing_bool:
                    sess.run(update_alpha)

                sess.run(ema_op)
                in_phase_step = sess.run(update_intra_phase_step)

                if metrics_summary_bool:
                    if args.calc_metrics:
                        # if verbose:
                        # print('Computing and writing metrics...')
                        metrics = save_metrics(writer, sess, npy_data_validation, gen_sample, batch_energy_val, batch_ang_val, args.metrics_batch_size,
                                               global_size, global_step, get_xy_dim(phase, args.start_shape),
                                               args.horovod, get_compute_metrics_dict(args), num_metric_samples,
                                               args.data_mean, args.data_stddev, verbose)

                        # Optuna pruning and return value:
                        last_fid = metrics['FID']
                        if trial is not None:
                            # Only for inter-trial parallelism: each worker has its own trial, so should report its own parameters. Otherwise, only rank 0 should report
                            if hyperparam_opt_inter_trial:
                                print(
                                    f"Trial: {trial.number}, Worker: {hvd.rank()}, Parameters: {trial.params}, global_step: {global_step}, FID: {last_fid}")
                            elif hvd.rank() == 0:
                                print(
                                    f"Trial: {trial.number}, Parameters: {trial.params}, global_step: {global_step}, FID: {last_fid}")

                        # If we use intra-trial parallelism, only worker 0 should report and prune, the others should 'just' return. For inter-trial parallelism, each workers should report an prune.
                        if hyperparam_opt_inter_trial or (hyperparam_opt_intra_trial and (hvd.rank() == 0)):
                            trial.report(metrics['FID'], global_step)
                            should_prune = trial.should_prune()
                            if (args.horovod and (hvd.rank() == 0)):
                                print(
                                    "Sending signal to other horovod workers that trial has been pruned and they should return")
                                MPI.COMM_WORLD.bcast(should_prune, root=0)
                            if should_prune:
                                raise optuna.TrialPruned()
                        elif hyperparam_opt_intra_trial and (hvd.rank() != 0):
                            should_prune = False
                            should_prune = MPI.COMM_WORLD.bcast(should_prune, root=0)
                            if should_prune:
                                print(
                                    f"Received signal from rank 0 that trial {trial.number} should be pruned. Returning...")
                                return last_fid

                if verbose:
                    if large_summary_bool:
                        if not hyperparam_opt_inter_trial:
                            print('Writing large summary...')
                        writer.add_summary(summary_s, global_step)
                        writer.add_summary(summary_s_val, global_step)
                        writer.add_summary(summary_l, global_step)
                    elif small_summary_bool:
                        if not hyperparam_opt_inter_trial:
                            print('Writing small summary...')
                        writer.add_summary(summary_s, global_step)
                        writer.add_summary(summary_s_val, global_step)
                    elif speed_measurement_bool:
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='img_s', simple_value=img_s)]),
                                           global_step)
                    if args.optuna_use_best_trial or args.optuna_ntrials == 1 or hyperparam_opt_intra_trial or normal_run:
                        print_summary_to_stdout(global_step, in_phase_step, img_s, local_img_s, d_loss, g_loss,
                                                d_lr_val, g_lr_val, alpha)

                # Is only executed once per phase, because the mixing_bool is then flipped to False
                if mixing_bool and (global_step >= ((phase - args.starting_phase)
                                                    * (args.mixing_nimg + args.stabilizing_nimg)
                                                    + args.mixing_nimg)):
                    mixing_bool = False
                    sess.run(assign_zero)
                    if verbose:
                        print(f"Begin stabilizing epochs in phase {phase}")

                if mixing_bool:
                    assert alpha.eval() >= 0

                # Break out of loop when phase is done
                if global_step >= (phase - args.starting_phase + 1) * (args.stabilizing_nimg + args.mixing_nimg):
                    break

            if verbose:
                print("\n\n\n End of phase.")

                # Save Session. First, update the generator with the weights stored in the expontial moving average, then store it.
                sess.run(ema_update_weights)
                saver = tf.train.Saver(var_list)
                print("Writing final checkpoint file: model_{phase}")
                saver.save(sess, os.path.join(logdir, f'model_{phase}'))

                # Final metric computation is not parallelized, because we want it to be computed on all samples from the test set, without duplicates.
                # Since the sampling is random for each worker, doing this with all workers could result in some samples being seen by multiple times, with others not being seen at all.
                # Set horovod to False explicitely, otherwise rank 0 will wait forever for a response from the other ranks.
                print(f"Computing final metrics for phase {phase} ...")
                if args.compute_metrics_test:
                    start_metrics_test = time.time()
                    metrics_test = save_metrics(None, sess, npy_data_test, gen_sample, batch_energy, batch_ang, args.metrics_batch_size,
                                                global_size, global_step, get_xy_dim(phase, args.start_shape), False,
                                                get_compute_metrics_dict(args), len(npy_data_test), args.data_mean,
                                                args.data_stddev, verbose)
                    end_metrics_test = time.time()
                    print(f"Computing metrics on test set took {end_metrics_test - start_metrics_test} seconds")
                    print("Test dataset metrics:")
                    print(metrics_test)
                if args.compute_metrics_validation:
                    start_metrics_val = time.time()
                    metrics_val = save_metrics(None, sess, npy_data_validation, gen_sample, batch_energy, batch_ang, args.metrics_batch_size,
                                               global_size, global_step, get_xy_dim(phase, args.start_shape), False,
                                               get_compute_metrics_dict(args), len(npy_data_validation), args.data_mean,
                                               args.data_stddev, verbose)
                    end_metrics_val = time.time()
                    print(f"Computing metrics on validation set took {end_metrics_val - start_metrics_val} seconds")
                    print("Validation dataset metrics:")
                    print(metrics_val)
                    # Overwrite the last fid
                    last_fid = metrics_val['FID']
                if args.compute_metrics_train:
                    start_metrics_train = time.time()
                    metrics_train = save_metrics(None, sess, npy_data_train, gen_sample, batch_energy, batch_ang, args.metrics_batch_size,
                                                 global_size, global_step, get_xy_dim(phase, args.start_shape), False,
                                                 get_compute_metrics_dict(args), len(npy_data_train), args.data_mean,
                                                 args.data_stddev, verbose)
                    end_metrics_train = time.time()
                    print(f"Computing metrics on training set took {end_metrics_train - start_metrics_train} seconds")
                    print("Training dataset metrics:")
                    print(metrics_train)

                if trial is not None:
                    # Only for inter-trial parallelism: each worker has its own trial, so should report its own parameters. Otherwise, only rank 0 should report
                    if hyperparam_opt_inter_trial:
                        print(
                            f"Trial: {trial.number}, Worker: {hvd.rank()}, Parameters: {trial.params}, global_step: {global_step}, FID: {last_fid}")
                    elif hvd.rank() == 0:
                        print(
                            f"Trial: {trial.number}, Parameters: {trial.params}, global_step: {global_step}, FID: {last_fid}")

            if args.ending_phase:
                if phase == args.ending_phase:
                    print("Reached final phase, breaking.")
                    break

    return last_fid