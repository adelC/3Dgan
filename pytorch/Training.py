from loaders.HDF5torch import HDF5generator
import numpy as np
import glob
import time
import torch
import torch.nn.functional as F
#from torchsummary import summary
from architectures.simple_1layer import CNNsimple

try:
    import cPickle as pickle
except ImportError:
    import pickle

def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)                                                                                                                                                             
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
          raise exception

def mape(pred, target):
   loss = torch.mean(torch.abs(target-pred)/target)
   return loss

def training():
   model = CNNsimple()
   model.cuda()
   #summary(model, input_size=(1, 25, 25))
   #criterion = torch.nn.L1Loss()
   criterion = mape
   optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001, alpha= 0.9)

   base_path = "/bigdata/shared/LCD/NewV1/" # fixed angle
   MODEL_STORE_PATH = "results/1layer_rmpsprop_100epochs/"
   safe_mkdir(MODEL_STORE_PATH)
   modelfile = 'model/simple_1layer_rmsprop.ckpt'
   safe_mkdir(MODEL_STORE_PATH + 'model/') 
   history = 'history/loss_history_simple_1layer_rmsprop.pkl'
   safe_mkdir(MODEL_STORE_PATH + 'history/')
   particle =['Ele'] # particle types                                                                                                      
   sample_path = []
   for p in particle:
     sample_path.append(base_path + p + 'Escan/*.h5')

   batch_size = 128
   train_ratio = 0.9
   shuffle=False
   num_epochs = 100

   n_classes = len(particle)
   # gather sample files for each type of particle                                                                                                
   class_files = [[]] * n_classes
   for i, class_path in enumerate(sample_path):
      class_files[i] = sorted(glob.glob(class_path))
   files_per_class = min([len(files) for files in class_files])

   # split the train, test, and validation files                                                                                                  
   # get lists of [[class1_file1, class2_file1], [class1_file2, class2_file2], ...]                                                               
   train_files = []
   test_files = []
   n_train = int(files_per_class * train_ratio)
   for i in range(files_per_class):
       new_files = []
       for j in range(n_classes):
           new_files.append(class_files[j][i])
           if i < n_train:
               train_files.append(new_files)
           else:
               test_files.append(new_files)
   
   init = time.time()
   loss_list=[]
   train_loss_list = []
   test_loss_list = []
   
   for epoch in range(num_epochs):
     init=time.time()
     # data generator for train files      
     train_loader = HDF5generator(train_files, batch_size=batch_size, shuffle=shuffle, num_events=180000) 
     total_batches = train_loader.total_batches
     for i, data in enumerate(train_loader):
       outputs = model(data['ECAL'])
       loss = criterion(outputs, data['target']) 
       loss_list.append(loss.item())
       #loss = torch.autograd.Variable(loss, requires_grad=True) 
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
      
       if (i + 1) % 100 == 0:
         print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_batches, loss.item()))

     train_loss_list.append(np.mean(loss_list))
     # Test the model
     test_loader = HDF5generator(test_files, batch_size=batch_size, shuffle=shuffle, num_events=20000)
     model.eval()
     with torch.no_grad():
       total = 0
       loss_list=[]
       for data in test_loader:
         outputs = model(data['ECAL'])
         loss = criterion(outputs, data['target'])
         loss_list.append(loss.item())
       test_loss_list.append(np.mean(loss_list))    
     print('Test loss of the model on the 20000 test images: {} %'.format((loss.item())))
     print('The epoch {} took {} seconds.'.format(epoch, time.time()-init))
     pickle.dump({'train': train_loss_list, 'test': test_loss_list}, open(MODEL_STORE_PATH + history, 'wb'))
     # Save the model and plot
     torch.save(model.state_dict(), MODEL_STORE_PATH + modelfile)

if __name__ == "__main__":
    training()
