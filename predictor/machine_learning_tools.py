import torch
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision    
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE
from config import CONF

from torch.multiprocessing import Pool, Process, set_start_method

class NeuralNetworks():
    
    data_driven_models = {}

    def __init__(self):
        pass

    # Dataset class for Pytorch.
    class PredictorDataset(TensorDataset):
        def __init__(self, data, target, transform=None):
            self.data = data
            self.target = target
            self.transform = transform
    
        def __getitem__(self, index):
            x = self.data[index]
            y = self.target[index]
            
            if self.transform:
                x = self.transform(x)
                
            return x, y
        
        def __len__(self):
            return len(self.data)

    class CNN(nn.Module):
        def __init__(self, params):
            super(NeuralNetworks.CNN, self).__init__()
            self.conv1 = nn.Conv2d(*params['conv1'])
            self.pool = nn.MaxPool2d(*params['pool'])
            self.conv2 = nn.Conv2d(*params['conv2'])
            self.fc1 = nn.Linear(*params['linear'])

        def forward(self, x):
            in_size = x.size(0)
            
            x = F.relu(self.pool(self.conv1(x)))
            x = F.relu(self.pool(self.conv2(x)))
            
            # Flatten the tensor.
            x = x.view(in_size, -1)
            x = self.fc1(x) # Softmax is already computed in the loss function.

            return x

    def create_models(self, data_processed, params):

        self.data_driven_models['cnn'] = self.CNN(params['cnn'])

    def normalize_data(self, data):
        data = np.array(data)
        data_shape = data.shape
        data_row = np.reshape(data, [1, data.size])[0]

        data_max = np.amax(data_row)
        data_min = np.amin(data_row)
        
        DIFF = data_max - data_min;  a, b = -1, 1;  norm_diff = b - a
            
        data_normalized = []
        for value in data_row:
            data_normalized.append(( norm_diff*(value - data_min) / DIFF ) + a)
        
        data_normalized = np.array(data_normalized)
        data_normalized = np.reshape(data_normalized, data_shape)

        return data_normalized.tolist()

    def device_setting(self, cuda_available):
        
        if cuda_available:
            device = torch.device("cuda:0")
            try:
                set_start_method('spawn')
            except RuntimeError:
                pass
        
        else:
            device = torch.device("cpu")
        
        return device

    def bearings_data_join(self, data_processed, processed_function, params):
        
        data = {}
        for key, bearing_data in data_processed[processed_function].items():
            if processed_function == 'health_assessment':
                if data == {}:
                    data['normal'] = [bearing_data['health_states']['normal']]
                    data['fast_degradation'] = [bearing_data['health_states']['fast_degradation']]
                else:
                    data['normal'].append(bearing_data['health_states']['normal'])
                    data['fast_degradation'].append(bearing_data['health_states']['fast_degradation'])
           
            elif processed_function == 'hht_marginal_spectrum':
                if data == {}:
                    data = [bearing_data]
                else:
                    data.append(bearing_data)

        return data

    def health_states_classes(self, health_states):
        
        # Classifing health state data.
        health_states_classes = []

        for normal_states, fast_degradation_states in zip(health_states['normal'], health_states['fast_degradation']):
            
            normal_qty = normal_states[1] - normal_states[0] + 1
            fast_degradation_qty = fast_degradation_states[1] - fast_degradation_states[0] + 1

            health_states_classes.extend([0]*normal_qty + [1]*fast_degradation_qty)

        return health_states_classes

    def predict(self, data_processed, models_params, predictor_params):

        cnn_path = os.getcwd() + '/predictor/data/Processed_Data/cnn_model/cnn_model'
        
        # Setting to use GPU processor if avaiable. Current set to False, GPU is too slow. Need to investigate.
        cuda_available = torch.cuda.is_available(); cuda_available=False
        device = self.device_setting(cuda_available)

        # Getting processed data from all bearings.
        health_states = self.bearings_data_join(data_processed, 'health_assessment', predictor_params)
        hht_data = self.bearings_data_join(data_processed, 'hht_marginal_spectrum', predictor_params)

        """ Convolutional Neural Network (CNN) """
        # Classifing health_states and making health_states 1D.
        health_states = self.health_states_classes(health_states)

        # Making hht_data files 1D and getting spectrum.
        hht_data = [file_data for bearing_data in hht_data for file_data in bearing_data]
        hht_data = [x[1] for x in hht_data]

        # Normalizing HHT data.
        for i, data in enumerate(hht_data):
            hht_data[i] = self.normalize_data(data)

        # Shuffling and splitting data into traning data and validation data.
        train_hht, eval_hht, train_health_states, eval_health_states = train_test_split(hht_data, health_states, test_size=0.3, random_state=42, shuffle=True)
        
        # Oversampling fast degradation class. Must be after data splitting. Otherwise, we could could end up with the same observation in both datasets.
        # Selecting fast degradation data.         
        sm = SMOTE(random_state=42, ratio=1.0)
        train_hht, train_health_states = sm.fit_sample(train_hht, train_health_states)

        # Test also undersampling!

        # Shuffling data again.
        random_index = np.random.permutation(len(train_hht))
        train_hht = np.array(train_hht)[random_index]
        train_health_states = np.array(train_health_states)[random_index]
        
        # Reshaping HHT data for CNN. Not mandatory.
        # shape = [batch_size, color_channels, height, width]. In this case we set a number of batch files to the number of files which we will process per epoch. Height and width are user defined. 'color_channels' would be the depth which in our case is just 1.
        train_hht = [np.reshape(x, [1, 32, 40]) for x in train_hht]
        eval_hht = [np.reshape(x, [1, 32, 40]) for x in eval_hht]
        
        # Creating tensors.
        if cuda_available: 
            train_hht, eval_hht = torch.cuda.FloatTensor(train_hht), torch.cuda.FloatTensor(eval_hht)
            train_health_states, eval_health_states = torch.cuda.FloatTensor(train_health_states), torch.cuda.FloatTensor(eval_health_states)
        else:
            train_hht, eval_hht = torch.FloatTensor(train_hht), torch.FloatTensor(eval_hht)
            train_health_states, eval_health_states = torch.LongTensor(train_health_states), torch.LongTensor(eval_health_states)

        # Creating dataset.
        train_dataset = self.PredictorDataset(train_hht, train_health_states)
        eval_dataset = self.PredictorDataset(eval_hht, eval_health_states)
        
        # Creating loaders.
        train_loader = DataLoader(
            train_dataset,
            batch_size=models_params['cnn_batch_size'],
            shuffle=True,
            num_workers=3,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=models_params['cnn_batch_size'],
            shuffle=True,
            num_workers=3,
        )

        # Passing model to device and setting criterion.
        model = self.data_driven_models['cnn'].to(device)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(models_params['cnn_epochs']):
            start = time.time()
            # Setting optimizer and adjusting learning rate with epochs.
            optimizer = optim.SGD(model.parameters(), lr=self.adjust_learning_rate(epoch))
            
            # Setting model to trainning mode.
            model.train()
            running_loss = 0.0
            for i, (hht_data, health_states) in enumerate(train_loader):
                
                # Passing data to device.
                hht_data, health_states = hht_data.to(device), health_states.to(device)
                
                # Setting the parameter gradients to zero.
                optimizer.zero_grad()

                # Forwarding data and getting prediction.
                health_states_ = model(hht_data)
                
                # Computing loss and ajusting CNN weights.
                loss = criterion(health_states_, health_states)
                loss.backward(loss)
                
                # Ajusting the weights of CNN.
                optimizer.step()
                
                # Averaging running_loss.
                running_loss = (running_loss*i + loss.item())/(i+1)
            
            if epoch % 1 == 0:
                print('Epoch', epoch)
                print('Loss Train %.4f' % running_loss)
                #print('Kappa', cohen_kappa_score())

            model.eval()
            running_loss = 0.0
            for i, (hht_data, health_states) in enumerate(eval_loader):

                # Passing data to device.
                hht_data, health_states = hht_data.to(device), health_states.to(device)

                # Forwarding data and getting prediction.
                health_states_ = model(hht_data)
                loss = criterion(health_states_, health_states)

                # Averaging running_loss.
                running_loss = (running_loss*i + loss.item())/(i+1)
                
            if epoch % 1 == 0:
                end = time.time()
                print('Loss Test %.4f' % running_loss)
                print('Time %.2f' % (end - start), '\n')
        
        # Saving model.
        torch.save(model.state_dict(), cnn_path)

    def adjust_learning_rate(self, epoch):
        lr = 0.001

        if epoch > 180:
            lr = lr / 1000000
        elif epoch > 140:
            lr = lr / 100000
        elif epoch > 110:
            lr = lr / 10000
        elif epoch > 80:
            lr = lr / 1000
        elif epoch > 50:
            lr = lr / 100
        elif epoch > 30:
            lr = lr / 10
        
        return lr






