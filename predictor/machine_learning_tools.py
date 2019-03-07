import torch
import time
import os
import pickle
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
    class ModelsDataset(TensorDataset):
        def __init__(self, data, target, transform=None):   
            self.data = data
            self.transform = transform
            self.target = target

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
            self.fc1 = nn.Linear(*params['linear1'])
            self.fc2 = nn.Linear(*params['linear2'])

        def forward(self, x):
            in_size = x.size(0)
            
            x = F.relu(self.pool(self.conv1(x)))
            x = F.relu(self.pool(self.conv2(x)))
            
            x = x.view(in_size, -1)              # Flatten the tensor.
            deep_features = F.relu(self.fc1(x))  # Saving fully connected layer output as deep_features.
            cnn_output = self.fc2(deep_features) # Softmax is already computed in the loss function.

            return [cnn_output, deep_features]

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

        # Setting to use GPU processor if avaiable. Current set to False, GPU is too slow. Need to investigate.
        device = self.device_setting(predictor_params['cuda_available'])

        """ Convolutional Neural Network (CNN) """
        # Train or load saved CNN model and pre-processed hht_data.
        self.cnn_train(data_processed, models_params, predictor_params, device)
        
        # Extract deep features.
        self.cnn_deep_features(data_processed, models_params, predictor_params, device)

    def cnn_deep_features(self, data_processed, models_params, predictor_params, device):
    
        deep_features_path = os.getcwd() + '/predictor/data/Processed_Data/cnn_model/deep_features'
        
        if(os.path.isfile(deep_features_path)):
            with open(deep_features_path, 'rb') as file:
                return pickle.load(file)

        cnn_model = self.data_driven_models['cnn']
        bearings_deep_features = []

        bearings_deep_feature_loader = self.cnn_data(data_processed, models_params, predictor_params, 'deep_feature_extraction')
        
        for bearing_deep_feature_loader in bearings_deep_feature_loader:
            bearing_deep_features = []
            import ipdb; ipdb.set_trace()
            for i, (eval_feature, _) in enumerate(bearing_deep_feature_loader):
                # Passing data to device.
                eval_feature = eval_feature.to(device)
            
                # Forwarding data and getting deep features. Appending numpy arrays because tensors uses more memory.
                bearing_deep_features.append([x.detach().numpy() for x in cnn_model(eval_feature)[1]])
                
                if i % 25 == 0:
                    print(model_params['cnn_batch_size']*(i+1), 'deep feature extracted.')

            bearings_deep_features.append(bearing_deep_features)

        # Saving pre-processed hht_data.
        with open(deep_features_path, 'wb') as file:
            pickle.dump(bearings_deep_features, file)
        
        data_processed['deep_features'] = bearings_deep_features

    def cnn_data(self, data_processed, models_params, predictor_params, adjust_data_for):

        deep_features_hht_data_path = os.getcwd() + '/predictor/data/Processed_Data/cnn_model/deep_feature_hht_data'

        if adjust_data_for == 'trainning':
            
            # Getting processed data from all bearings.
            health_states = self.bearings_data_join(data_processed, 'health_assessment', predictor_params)
            hht_data = self.bearings_data_join(data_processed, 'hht_marginal_spectrum', predictor_params)
            
            # Classifing health_states and making health_states 1D.
            health_states = self.health_states_classes(health_states)
            
            # Saving hht_data shape to reshape for deep features.
            hht_data_shapes = [len(x) for x in hht_data]
            index_slices = []; index_slice = 0
            for shape in enumerate(hht_data_shapes):
                index_slice += shape[1]
                index_slices.append(index_slice) 

            # Making hht_data files 1D and getting spectrum.
            hht_data = [file_data[1] for bearing_data in hht_data for file_data in bearing_data]

            # Normalizing HHT data.
            for i, data in enumerate(hht_data):
                hht_data[i] = self.normalize_data(data)

            # Saving hht_data at this point for deep features.
            with open(deep_features_hht_data_path, 'wb') as file:
                pickle.dump(np.split(hht_data, index_slices[:-1]), file)

            # Splitting data into traning data and validation data.
            train_hht, eval_hht, train_health_states, eval_health_states = train_test_split(hht_data, health_states, test_size=0.3, random_state=14, shuffle=True)

            # Oversampling fast degradation class. Must be after data splitting. Otherwise, we could could end up with the same observation in both datasets.
            # Selecting fast degradation data.         
            sm = SMOTE(random_state=42, ratio=1.0)
            train_hht, train_health_states = sm.fit_sample(train_hht, train_health_states)
            
            # Shuffling data.
            random_index = np.random.permutation(len(train_hht))
            train_hht = np.array(train_hht)[random_index]
            train_health_states = np.array(train_health_states)[random_index]

            train_hht = [np.reshape(x, [1, 32, 40]) for x in train_hht]
            eval_hht = [np.reshape(x, [1, 32, 40]) for x in eval_hht]

            # Creating tensors.
            if predictor_params['cuda_available']: 
                train_hht, eval_hht = torch.cuda.FloatTensor(train_hht), torch.cuda.FloatTensor(eval_hht)
                train_health_states, eval_health_states = torch.cuda.LongTensor(train_health_states), torch.cuda.LongTensor(eval_health_states)
            else:
                train_hht, eval_hht = torch.FloatTensor(train_hht), torch.FloatTensor(eval_hht)
                train_health_states, eval_health_states = torch.LongTensor(train_health_states), torch.LongTensor(eval_health_states)

            # Creating dataset.
            train_dataset = self.ModelsDataset(train_hht, train_health_states)
            eval_dataset = self.ModelsDataset(eval_hht, eval_health_states)
            
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

            return train_loader, eval_loader

        elif adjust_data_for == 'deep_feature_extraction':
           
           # Loading saved deep_features_hht_data.
            if(os.path.isfile(deep_features_hht_data_path)):
                with open(deep_features_hht_data_path, 'rb') as file:
                    bearings_hht_data = pickle.load(file)

            # Reshaping bearings_hht_data.
            bearings_hht_data_reshaped = []
            for bearing_hht_data in bearings_hht_data:
                bearings_hht_data_reshaped.append([np.reshape(x, [1, 32, 40]) for x in bearing_hht_data])

            # Creating tensor for hht_data.
            bearings_hht_tensor = []
            for bearing_hht_data in bearings_hht_data_reshaped:
                if predictor_params['cuda_available']: 
                    bearings_hht_tensor.append(torch.cuda.FloatTensor(bearing_hht_data))
                else:
                    bearings_hht_tensor.append(torch.FloatTensor(bearing_hht_data))
            
            bearings_deep_feature_dataset = []
            for bearing_hht_tensor in bearings_hht_tensor:
                N = len(bearing_hht_tensor)
                bearings_deep_feature_dataset.append(self.ModelsDataset(bearing_hht_tensor, [-1]*N)) # No need data for output. This dataset is just for forward process.

            bearings_deep_feature_loader = []
            for bearing_deep_feature_dataset in bearings_deep_feature_dataset:
                bearings_deep_feature_loader.append(
                DataLoader(
                    bearing_deep_feature_dataset,
                    batch_size=models_params['cnn_batch_size'],
                    shuffle=True,
                    num_workers=3,
                ))

            return bearings_deep_feature_loader

    def cnn_train(self, data_processed, models_params, predictor_params, device): 
        
        cnn_path = os.getcwd() + '/predictor/data/Processed_Data/cnn_model/cnn_model'

        # Loading saved model and hht_data pre-processed.
        if(os.path.isfile(cnn_path)):
            self.data_driven_models['cnn'].load_state_dict(torch.load(cnn_path))
            return
            
        # Getting data loaders.
        train_loader, eval_loader = self.cnn_data(data_processed, models_params, predictor_params, 'trainning')

        # Passing model to device and setting criterion.
        cnn_model = self.data_driven_models['cnn'].to(device)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(models_params['cnn_epochs']):
            
            if epoch % 1 == 0:
                start = time.time()
            
            # Setting optimizer and adjusting learning rate with epochs.
            optimizer = optim.SGD(cnn_model.parameters(), lr=self.adjust_learning_rate(epoch))
            
            # Setting model to trainning mode.
            cnn_model.train()
            running_loss = 0.0
            
            for i, (train_feature, train_output) in enumerate(train_loader):
                
                # Passing data to device.
                train_feature, train_output = train_feature.to(device), train_output.to(device)
                
                # Setting the parameter gradients to zero.
                optimizer.zero_grad()

                # Forwarding data and getting prediction.
                train_output_ = cnn_model(train_feature)[0]
                
                # Computing loss and ajusting CNN weights.
                loss = criterion(train_output_, train_output)
                loss.backward(loss)
                
                # Ajusting the weights of CNN.
                optimizer.step()
                
                # Averaging running_loss.
                running_loss = (running_loss*i + loss.item())/(i+1)
            
            if epoch % 1 == 0:
                print('Epoch', epoch)
                print('Loss Train %.4f' % running_loss)
                
                # Try to calculate kappa score here.

            cnn_model.eval()
            running_loss = 0.0
            
            for i, (eval_feature, eval_output) in enumerate(eval_loader):

                # Passing data to device.
                eval_feature, eval_output = eval_feature.to(device), eval_output.to(device)

                # Forwarding data and getting prediction.
                eval_output_ = cnn_model(eval_feature)[0]
                loss = criterion(eval_output_, eval_output)

                # Averaging running_loss.
                running_loss = (running_loss*i + loss.item())/(i+1)
                
            if epoch % 1 == 0:
                end = time.time()
                print('Loss Test %.4f' % running_loss)
                print('Time %.2f' % (end - start), 's\n')
        
        # Saving model.
        torch.save(cnn_model.state_dict(), cnn_path)

    def adjust_learning_rate(self, epoch):
        lr = 0.001

        if epoch > 80:
            lr = lr / 10000
        elif epoch > 60:
            lr = lr / 1000
        elif epoch > 50:
            lr = lr / 100
        elif epoch > 30:
            lr = lr / 10
        
        return lr
