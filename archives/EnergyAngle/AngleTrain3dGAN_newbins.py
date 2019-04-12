#Training for GAN
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import keras
import argparse
import os
os.environ['LD_LIBRARY_PATH'] = os.getcwd()
from six.moves import range
import sys
import h5py 
import numpy as np
import time
import math
import argparse

if os.environ.get('HOSTNAME') == 'tlab-gpu-gtx1080ti-06.cern.ch': # Here a check for host can be used
    tlab = True
else:
    tlab= False
    
try:
    import setGPU #if Caltech
except:
    pass

#from memory_profiler import profile # used for memory profiling
import keras.backend as K
import analysis.utils.GANutils as gan
#K.set_image_dim_ordering('tf')

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils.generic_utils import Progbar
#config = tf.ConfigProto(log_device_placement=True)

# printing versions of software used
#print('keras version:', keras.__version__)
#print('python version:', sys.version)
#import tensorflow as tf
#print('tensorflow version', tf.__version__)
#print('numpy version', np.version.version)

def main():
    #Architectures to import
    if keras.__version__ == '1.2.2':
        from AngleArch3dGAN_newbins import generator, discriminator
    else:
        from AngleArch3dGAN_k2 import generator, discriminator

    #Values to be set by user
    parser = get_parser()
    params = parser.parse_args()
    nb_epochs = params.nbepochs      #Total Epochs for training
    batch_size = params.batchsize    #batch size
    latent_size = params.latentsize  #latent vector size
    verbose = params.verbose
    datapath = params.datapath       #Data path
    nEvents = params.nEvents         #Maximum limit on data entries 
    ascale = params.ascale           #scaling used for angle
    yscale = params.yscale           #scaling of primary energy = 1/yscale
    weightdir = 'weights/3dgan_weights_' + params.name           #dir for weight storage
    pklfile = 'results/3dgan_history_' + params.name + '.pkl'    # loss history
    resultfile = 'results/3dgan_analysis' + params.name + '.pkl' # optimization metric history
    xscale = params.xscale
    xpower = params.xpower
    analyse=params.analyse           # if analysing
    energies =params.energies        # Bins
    loss_weights=[params.gen_weight, params.aux_weight, params.ang_weight, params.ecal_weight, params.hist_weight]
    
    thresh = params.thresh # threshold for data
    angtype = params.angtype

    if tlab:
      datapath = '/gkhattak/*Measured3ThetaEscan/*.h5'
      weightdir = '/gkhattak/weights/3Dweights'
      pklfile = '/gkhattak/results/3dgan_history.pkl'

    print(params)

    # Building discriminator and generator
    gan.safe_mkdir(weightdir)
    d=discriminator(xpower)
    g=generator(latent_size)
    Gan3DTrainAngle(d, g, datapath, nEvents, weightdir, pklfile, nb_epochs=nb_epochs, batch_size=batch_size,
                    latent_size=latent_size, loss_weights=loss_weights, xscale = xscale, xpower=xpower, angscale=ascale,
                    yscale=yscale, thresh=thresh, angtype=angtype, analyse=analyse, resultfile=resultfile,
                    energies=energies)

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--nbepochs', action='store', type=int, default=1, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=256, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/data/shared/gkhattak/*Measured3ThetaEscan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--nEvents', action='store', type=int, default=200000, help='Maximum Number of events used for Training')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--xscale', action='store', type=int, default=1, help='Multiplication factor for ecal deposition')
    parser.add_argument('--xpower', action='store', type=float, default=0.75, help='pre processing of cell energies by raising to a power')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--ascale', action='store', type=int, default=1, help='Multiplication factor for angle input')
    parser.add_argument('--analyse', action='store', default=True, help='Whether or not to perform analysis')
    parser.add_argument('--energies', action='store', type=int, default=[0, 110, 150, 190], help='Energy bins for analysis')
    parser.add_argument('--gen_weight', action='store', type=float, default=3, help='loss weight for generation real/fake loss')
    parser.add_argument('--aux_weight', action='store', type=float, default=0.1, help='loss weight for auxilliary energy regression loss')
    parser.add_argument('--ang_weight', action='store', type=float, default=25, help='loss weight for angle loss')
    parser.add_argument('--ecal_weight', action='store', type=float, default=0.1, help='loss weight for ecal sum loss')
    parser.add_argument('--hist_weight', action='store', type=float, default=0.1, help='loss weight for additional bin count loss')
    parser.add_argument('--thresh', action='store', type=int, default=0., help='Threshold for cell energies')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle to use for Training. It can be theta, mtheta or eta')
    parser.add_argument('--name', action='store', type=str, default='', help='Identifier for training.')
    return parser

# A histogram fucntion that counts cells in different bins
def hist_count(x, p=1):
    bin1 = np.sum(np.where(x>(0.08**p) , 1, 0), axis=(1, 2, 3))
    bin2 = np.sum(np.where((x<(0.08**p)) & (x>(0.05**p)), 1, 0), axis=(1, 2, 3))
    bin3 = np.sum(np.where((x<(0.05**p)) & (x>(0.035**p)), 1, 0), axis=(1, 2, 3))
    bin4 = np.sum(np.where((x<(0.035**p)) & (x>(0.025**p)), 1, 0), axis=(1, 2, 3))
    bin5 = np.sum(np.where((x<(0.025**p)) & (x>(0.018**p)), 1, 0), axis=(1, 2, 3))
    bin6 = np.sum(np.where((x<(0.018**p)) & (x>(0.0122**p)), 1, 0), axis=(1, 2, 3))
    bin7 = np.sum(np.where((x<(0.0122**p)) & (x>(0.009**p)), 1, 0), axis=(1, 2, 3))
    bin8 = np.sum(np.where((x<(0.009**p)) & (x>(0.006**p)), 1, 0), axis=(1, 2, 3))    
    bin9 = np.sum(np.where((x<(0.006**p)) & (x>(0.0025**p)), 1, 0), axis=(1, 2, 3))
    bin10 = np.sum(np.where((x<(0.0025**p)) & (x>0.), 1, 0), axis=(1, 2, 3))
    bin11 = np.sum(np.where(x==0, 1, 0), axis=(1, 2, 3))
    bins = np.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10, bin11], axis=1)
    bins[np.where(bins==0)]=1 # so that an empty bin will be assigned a count of 1 to avoid unstability
    return bins

#get data for training
def GetDataAngle(datafile, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4):
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    ang = np.array(f.get(angtype))
    X=np.array(f.get('ECAL'))* xscale
    Y=np.array(f.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ang = ang.astype(np.float32)
    X = np.expand_dims(X, axis=-1)
    ecal = np.sum(X, axis=(1, 2, 3))
    if xpower !=1.:
        X = np.power(X, xpower)
    return X, Y, ang, ecal

def Gan3DTrainAngle(discriminator, generator, datapath, nEvents, WeightsDir, pklfile, nb_epochs=30, batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1., xpower=1., angscale=1., angtype='theta', yscale=100., thresh=0., analyse=False, resultfile="", energies=[]):
    start_init = time.time()
    verbose = False    
    particle='Ele'
    f = [0.9, 0.1]         # train test fractions
    loss_ftn = hist_count  # function used for additional loss

    # building discriminator
    print('[INFO] Building discriminator')
    discriminator.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )

    # build the generator
    print('[INFO] Building generator')
    generator.compile(
        optimizer=RMSprop(),
        loss='binary_crossentropy'
    )
 
    # build combined Model
    latent = Input(shape=(latent_size, ), name='combined_z')   
    fake_image = generator( latent)
    discriminator.trainable = False
    fake, aux, ang, ecal, add_loss= discriminator(fake_image)
    combined = Model(
        input=[latent],
        output=[fake, aux, ang, ecal, add_loss],
        name='combined_model'
    )
    combined.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )

    # Getting All available Data sorted in test train fraction
    Trainfiles, Testfiles = gan.DivideFiles(datapath, f, datasetnames=["ECAL"], Particles =[particle])

    # Printing all available files for train and test but the actual batches will also depend on nEvents
    print(Trainfiles)
    print(Testfiles)
    Trainfiles=Trainfiles[:1]
    Testfiles =Testfiles[:1]
    
    # Maximum number of events used according to nEvents
    nb_Test = int(nEvents * f[1]) # The number of test events calculated from fraction of nEvents
    nb_Train = int(nEvents * f[0]) # The number of train events calculated from fraction of nEvents

    # The number of maximum possible batches.
    #The actual number of batches used will be limited by the min of nb_train_batches and available data
    nb_train_batches = int(nb_Train/batch_size)
    nb_test_batches = int(nb_Test/batch_size)
    print('The max train batches can be {} batches while max test batches can be {}'.format(nb_train_batches, nb_test_batches))  

    # dict for loss history
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    init_time = time.time()- start_init
    analysis_history = defaultdict(list)
    print('Initialization time is {} seconds'.format(init_time))
    
    ################################################ Train ##########################################################################

    # Repeat for number of epochs
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        #read first train file
        X_train, Y_train, ang_train, ecal_train = GetDataAngle(Trainfiles[0], xscale=xscale,
                                                xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
        nb_file=1    # The file number
        nb_batches = int(X_train.shape[0] / batch_size)  #batches loaded
        epoch_gen_loss = []
        epoch_disc_loss = []
        index = 0    #counter for batch in epoch
        file_index=0 #counter for batch within a file

        #Repeat till the limit of available data or maximum number of batches is reached
        while ((nb_file < len(Trainfiles)) or (nb_batches > 0)) and index < nb_train_batches:
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    print('processed {} batches'.format(index + 1))
            #check if all loaded bacthes are processed
            if nb_batches==0:
                #remove processed data
                X_train = X_train[(file_index * batch_size):]
                Y_train = Y_train[(file_index * batch_size):]
                ang_train = ang_train[(file_index * batch_size):]
                ecal_train = ecal_train[(file_index * batch_size):]
                #read new file into temp                                                
                X_temp, Y_temp, ang_temp, ecal_temp = GetDataAngle(Trainfiles[nb_file], xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
                nb_file+=1
                # concat to previous data
                X_train = np.concatenate((X_train, X_temp))
                Y_train = np.concatenate((Y_train, Y_temp))
                ang_train = np.concatenate((ang_train, ang_temp))
                ecal_train = np.concatenate((ecal_train, ecal_temp))
                nb_batches = int(X_train.shape[0] / batch_size)
                del X_temp, Y_temp, ang_temp, ecal_temp
                
                print("{} batches loaded..........".format(nb_batches))
                file_index = 0
                
            #loade a batch of data
            image_batch = X_train[(file_index * batch_size):(file_index  + 1) * batch_size]
            energy_batch = Y_train[(file_index * batch_size):(file_index + 1) * batch_size]
            ecal_batch = ecal_train[(file_index *  batch_size):(file_index + 1) * batch_size]
            ang_batch = ang_train[(file_index * batch_size):(file_index + 1) * batch_size]
            add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower), axis=-1)
            file_index +=1
            nb_batches -=1
            #generate an image
            noise = np.random.normal(0, 1, (batch_size, latent_size-1))
            noise = np.multiply(energy_batch.reshape(-1, 1), noise) # Same energy as G4
            generator_ip = np.concatenate((ang_batch.reshape(-1, 1), noise), axis=1)
            generated_images = generator.predict(generator_ip, verbose=0)
            #train discriminator on data image
            real_batch_loss = discriminator.train_on_batch(image_batch, [gan.BitFlip(np.ones(batch_size)), energy_batch, ang_batch, ecal_batch, add_loss_batch])
            #train discriminator on generated image
            fake_batch_loss = discriminator.train_on_batch(generated_images, [gan.BitFlip(np.zeros(batch_size)), energy_batch, ang_batch, ecal_batch, add_loss_batch])

            #early stopping if generating empty images
            #if ecal sum has 100% loss then end the training
            if fake_batch_loss[3] == 100.0 and index >10:
                print("Empty image with Ecal loss equal to 100.0 for {} batch".format(index))
                generator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(g_weights), overwrite=True)
                discriminator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(d_weights), overwrite=True)
                print ('real_batch_loss', real_batch_loss)
                print ('fake_batch_loss', fake_batch_loss)
                sys.exit()
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])
            trick = np.ones(batch_size)
            gen_losses = []
            #train generator twice
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size-1))
                noise = np.multiply(energy_batch.reshape(-1, 1), noise)
                generator_ip = np.concatenate((ang_batch.reshape(-1, 1), noise), axis=1) # sampled angle same as g4 theta
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, energy_batch.reshape(-1, 1), ang_batch, ecal_batch, add_loss_batch]))
            generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]
            epoch_gen_loss.append(generator_loss)
            index +=1

            # Used at design time for debugging
            #print('real_batch_loss', real_batch_loss)
            #print ('fake_batch_loss', fake_batch_loss)
            #disc_out = discriminator.predict(image_batch)
            #print('disc_out')
            #print(np.transpose(disc_out[4][:20].astype(int)))
            #print('add_loss_batch')
            #print(np.transpose(add_loss_batch[:20]))

        # Testing
        # Test process will also be accomplished in batches to reduce memory consumption
        print ('Total batches were {}'.format(index))
        print('Time taken by epoch{} was {} seconds.'.format(epoch, time.time()-epoch_start))
        print('\nTesting for epoch {}:'.format(epoch))
        test_start = time.time()

        ####################################  Test  ##################################################################

        #read first test file
        X_test, Y_test, ang_test, ecal_test = GetDataAngle(Testfiles[0], xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
        nb_batches = int(X_train.shape[0] / batch_size)
        disc_test_loss=[]
        gen_test_loss =[]
        nb_file=1
        index=0
        file_index=0
        while ((nb_file < len(Testfiles)) or (nb_batches > 0)) and index < nb_test_batches:
           if nb_batches==0:
               X_test = X_test[(file_index * batch_size):]
               Y_test = Y_test[(file_index * batch_size):]
               ang_test = ang_test[(file_index * batch_size):]
               ecal_test = ecal_test[(file_index * batch_size):]
               X_temp, Y_temp, ang_temp, ecal_temp = GetDataAngle(Testfiles[nb_file], xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
               nb_file+=1

               X_test = np.concatenate((X_test, X_temp))
               Y_test = np.concatenate((Y_test, Y_temp))
               ang_test = np.concatenate((ang_test, ang_temp))
               ecal_test = np.concatenate((ecal_test, ecal_temp))
               nb_batches = int(X_train.shape[0] / batch_size)

               print("{} test batches loaded..........".format(nb_batches))
               file_index = 0
           image_batch = X_test[(file_index * batch_size):(file_index  + 1) * batch_size]
           energy_batch = Y_test[(file_index * batch_size):(file_index + 1) * batch_size]
           ecal_batch = ecal_test[(file_index *  batch_size):(file_index + 1) * batch_size]
           ang_batch = ang_test[(file_index * batch_size):(file_index + 1) * batch_size]
           add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower), axis=-1)
           file_index +=1
           nb_batches -=1
           noise = np.random.normal(0, 1, (batch_size, latent_size-1))
           noise = np.multiply(energy_batch.reshape(-1, 1), noise)
           generator_ip = np.concatenate((ang_batch.reshape(-1, 1), noise), axis=1)
           generated_images = generator.predict(generator_ip, verbose=False)
           add_loss_test = loss_ftn(image_batch, xpower)
           X = np.concatenate((image_batch, generated_images))
           y = np.array([1] * batch_size + [0] * batch_size)
           ang = np.concatenate((ang_batch, ang_batch))
           ecal = np.concatenate((ecal_batch, ecal_batch))
           aux_y = np.concatenate((energy_batch, energy_batch), axis=0)
           add_loss= np.concatenate((add_loss_batch, add_loss_batch), axis=0)
           index +=1           
           disc_test_loss.append(discriminator.evaluate( X, [y, aux_y, ang, ecal, add_loss], verbose=False, batch_size=batch_size))
           gen_test_loss.append(combined.evaluate(generator_ip,
                    [np.ones(batch_size), energy_batch, ang_batch, ecal_batch, add_loss_batch]
                    , verbose=False, batch_size=batch_size))
        print('Total Test batches were {}'.format(index))
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        discriminator_test_loss = np.mean(np.array(disc_test_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        generator_test_loss = np.mean(np.array(gen_test_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        
        ####################### Short Analysis for HP scan ##############################################

        # If also analyzing then entire test data will be loaded
        if analyse:
            var = gan.get_sorted_angle(Testfiles, energies, angtype=angtype, thresh=thresh)
            result = gan.AngleAnalysisShort(var, generator, energies, latent_size)
            print('Analysing............')
            analysis_history['total'].append(result[0])
            analysis_history['energy'].append(result[1])
            analysis_history['moment'].append(result[2])
            analysis_history['angle'].append(result[3])
            print('Result = ', result)
            pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

        print('{0:<20s} | {1:6s} | {2:12s} | {3:12s}| {4:5s} | {5:8s} | {6:8s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)
        ROW_FMT = '{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}| {6:<10.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
        discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                   overwrite=True)
        
        epoch_time = time.time()-test_start
        print("The Testing for {} epoch took {} seconds. Weights are saved in {}".format(epoch, epoch_time, WeightsDir))
        pickle.dump({'train': train_history, 'test': test_history}, open(pklfile, 'wb'))

if __name__ == '__main__':
    main()