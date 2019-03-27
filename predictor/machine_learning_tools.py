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
from config import CONF
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
from mpl_toolkits.mplot3d import Axes3D
from imblearn.under_sampling import RandomUnderSampler

from torch.multiprocessing import Pool, Process, set_start_method

torch.multiprocessing.set_sharing_strategy('file_system')

# Fix random seed for reproducibility
np.random.seed(7)

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
            
            x = x.view(in_size, -1)                       # Flatten the tensor.
            deep_features = self.fc1(x)                   # Saving first linear layer output as deep_features.
            cnn_output = self.fc2(F.relu(deep_features))  # Softmax is already computed in the loss function.

            return [cnn_output, deep_features]

    class LSTM(nn.Module):
       
        def __init__(self, params):
            super(NeuralNetworks.LSTM, self).__init__()

            self.lstm = nn.LSTM(input_size=params['input'], hidden_size=params['hidden'], num_layers=params['num_layers'])
            self.fc = nn.Linear(*params['linear'])    

        def init_hidden(self, params):
            self.h_t = torch.zeros(params['num_layers'], params['batch_size'], params['hidden'])
            self.h_c = torch.zeros(params['num_layers'], params['batch_size'], params['hidden'])

        def forward(self, x):
            lstm_outs, (self.h_t, self.h_c) = self.lstm(x, (self.h_t, self.h_c))
            
            # Getting the last output.
            return self.fc(lstm_outs[-1]) 
            
    def create_models(self, data_processed, params):
        
        self.data_driven_models['cnn'] = self.CNN(params['cnn'])
        self.data_driven_models['lstm'] = self.LSTM(params['lstm'])

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
        for _, bearing_data in data_processed[processed_function].items():
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
            elif processed_function == 'bearings_fft':
                if data == {}:
                    data = [bearing_data]
                else:
                    data.append(bearing_data)
        return data

    def health_states_classes(self, health_states):
        
        # Classifing health state data. '0' for normal and '1' for degradate.
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

        # Train or load saved LSTM model.
        self.lstm_train(data_processed, models_params, predictor_params, device)

    def cnn_deep_features(self, data_processed, models_params, predictor_params, device):
    
        deep_features_path = os.getcwd() + '/predictor/data/Processed_Data/cnn_model/deep_features'
        
        if os.path.isfile(deep_features_path):
            with open(deep_features_path, 'rb') as file:
                data_processed['deep_features'] = pickle.load(file)
                return

        # Getting the CNN model.
        cnn_model = self.data_driven_models['cnn']
        cnn_model.eval()
        bearings_deep_features = []

        # Getting data to extract deep features.
        bearings_deep_feature_loader = self.cnn_data(data_processed, models_params, predictor_params, 'deep_feature_extraction')
   
        for bearing_deep_feature_loader in bearings_deep_feature_loader:
            bearing_deep_features = []
            for i, (eval_feature, eval_output) in enumerate(bearing_deep_feature_loader):

                # Passing data to device.
                eval_feature, eval_output = eval_feature.to(device), eval_output.to(device)

                # Forwarding data and getting deep features. Appending numpy arrays because tensors uses more memory.
                bearing_deep_features.extend([x.detach().numpy() for x in cnn_model(eval_feature)[1]])

                if i % models_params['cnn_batch_size'] == 0:
                    print(models_params['cnn_batch_size']*(i+1), 'deep feature extracted.')
            
            # Standarizing deep features.
            bearing_deep_features = StandardScaler().fit_transform(bearing_deep_features)

            # Calculating PCA of deep features and plotting.        
            pca = PCA(n_components=3, svd_solver='full')
            target = np.array(bearing_deep_feature_loader.dataset.target)
            _, counts = np.unique(target, return_counts=True)

            pca_components_normal = pca.fit_transform(bearing_deep_features[0:counts[0]])
            pca_components_degradation = pca.fit_transform(bearing_deep_features[counts[0] + 1:counts[0] + counts[1]])
            
            fig = plt.figure()
            
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pca_components_normal[:, 0], pca_components_normal[:, 1], pca_components_normal[:, 2], color='#509a2d')
            ax.scatter(pca_components_degradation[:, 0], pca_components_degradation[:, 1], pca_components_degradation[:, 2], color='#ed1d24')

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            #plt.show()
            bearings_deep_features.append(np.array(bearing_deep_features))

        # Saving deep features.
        with open(deep_features_path, 'wb') as file:
            pickle.dump(bearings_deep_features, file)
        
        data_processed['deep_features'] = np.array(bearings_deep_features)

    def normalize_data(self, data, interval=[-1,1]):
     
        data = np.array(data)
        data_shape = data.shape
        data_row = np.reshape(data, [1, data.size])[0]

        data_max = np.amax(data_row)
        data_min = np.amin(data_row)
        
        DIFF = data_max - data_min;  a, b = interval;  norm_diff = b - a
            
        data_normalized = []
        for value in data_row:
            data_normalized.append(( norm_diff*(value - data_min) / DIFF ) + a)
        
        data_normalized = np.array(data_normalized)
        data_normalized = np.reshape(data_normalized, data_shape)

        return data_normalized.tolist()

    def cnn_data(self, data_processed, models_params, predictor_params, adjust_data_for):

        deep_features_hht_data_path = os.getcwd() + '/predictor/data/Processed_Data/cnn_model/deep_feature_hht_data'

        if adjust_data_for == 'trainning':
            cnn_data_trainning_path = os.getcwd() + '/predictor/data/Processed_Data/cnn_model/trainning_data'

            #if os.path.isfile(cnn_data_trainning_path):
            #    with open(cnn_data_trainning_path, 'rb') as file:
            #        return pickle.load(file)
            
            # Getting processed data from all bearings.
            health_states = self.bearings_data_join(data_processed, 'health_assessment', predictor_params)
            hht_data = self.bearings_data_join(data_processed, 'hht_marginal_spectrum', predictor_params)

            if models_params['cnn_data_type'] == 'imfs_hilbert_spectrum':
                for i, bearing_data in enumerate(hht_data):
                    hht_data[i] = [bearing_data[0], bearing_data[2]]
            

            # Classifing health_states and making health_states 1D.
            health_states = self.health_states_classes(health_states)
            
            # Saving hht_data shape to reshape for deep features.
            hht_data_shapes = [len(x[1]) for x in hht_data]
            index_slices = []; index_slice = 0
            for shape in enumerate(hht_data_shapes):
                index_slice += shape[1]
                index_slices.append(index_slice) 
            
            # Making hht_data files 1D and getting spectrum.
            hht_data = [files_hht for bearing_data in hht_data for files_hht in bearing_data[1]] 
    
            # Scaling data [-1, 1].
            hht_data = self.normalize_data(hht_data)

            # Saving hht_data at this point for deep features.
            with open(deep_features_hht_data_path, 'wb') as file:
                pickle.dump([np.split(hht_data, index_slices[:-1]), np.split(health_states, index_slices[:-1])], file)
            # Splitting data into traning data and validation data.
            train_hht, eval_hht, train_health_states, eval_health_states = train_test_split(hht_data, health_states, test_size=0.3, random_state=42, shuffle=False)
            
            """ Here we have a imbalanced dataset. It's important to do something about it here. In the next lines, oversampling and/or undersampling are explored."""
            # Maybe undersampling it's better because the normal samples are more similar between them than the degradated samples.
            # Undersampling normal class. Must be after data splitting. Otherwise, we could could end up with the same observation in both datasets.
            rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
            train_hht, train_health_states = rus.fit_resample(train_hht, train_health_states)
            
            # Reshaping data to use as CNN input. Not mandatory but this is done is the paper that I was trying to reproduce.
            train_hht = [np.reshape(x, predictor_params['hht_cnn_shape']) for x in train_hht]
            eval_hht = [np.reshape(x, predictor_params['hht_cnn_shape']) for x in eval_hht]

            # Creating tensors.
            if predictor_params['cuda_available']: 
                train_hht, eval_hht = torch.cuda.FloatTensor(train_hht), torch.cuda.FloatTensor(eval_hht)
                train_health_states, eval_health_states = torch.cuda.LongTensor(train_health_states), torch.cuda.LongTensor(eval_health_states)
            else:
                train_hht, eval_hht = torch.FloatTensor(train_hht), torch.FloatTensor(eval_hht)
                train_health_states, eval_health_states = torch.LongTensor(train_health_states), torch.LongTensor(eval_health_states)

            # Creating datasets.
            train_dataset = self.ModelsDataset(train_hht, train_health_states)
            eval_dataset = self.ModelsDataset(eval_hht, eval_health_states)
            
            # Creating data loaders.
            train_loader = DataLoader(
                train_dataset,
                batch_size=models_params['cnn_batch_size'],
                shuffle=True,
                num_workers=4,
            )

            eval_loader = DataLoader(
                eval_dataset,
                batch_size=models_params['cnn_batch_size'],
                shuffle=True,
                num_workers=4,
            )

            # Saving CNN trainning loaders.
            with open(cnn_data_trainning_path, 'wb') as file:
                pickle.dump((train_loader, eval_loader), file)

            return train_loader, eval_loader

        elif adjust_data_for == 'deep_feature_extraction':
            deep_features_loader_path = os.getcwd() + '/predictor/data/Processed_Data/cnn_model/deep_features_loader'

            if os.path.isfile(deep_features_loader_path):
                with open(deep_features_loader_path, 'rb') as file:
                    return pickle.load(file)

           # Loading saved deep_features_hht_data.
            if(os.path.isfile(deep_features_hht_data_path)):
                with open(deep_features_hht_data_path, 'rb') as file:
                    data = pickle.load(file)

            bearings_hht_data, health_states = data

            # Reshaping bearings_hht_data.
            bearings_hht_data_reshaped = []
            for bearing_hht_data in bearings_hht_data:
                bearings_hht_data_reshaped.append([np.reshape(x, predictor_params['hht_cnn_shape']) for x in bearing_hht_data])

            # Creating tensor for hht_data.
            bearings_hht_tensor = []; bearings_heatlh_states_tensor = []
            for bearing_hht_data, health_state in zip(bearings_hht_data_reshaped, health_states):
                if predictor_params['cuda_available']: 
                    bearings_hht_tensor.append(torch.cuda.FloatTensor(bearing_hht_data))
                    bearings_heatlh_states_tensor.append(torch.cuda.LongTensor(health_state))
                else:
                    bearings_hht_tensor.append(torch.FloatTensor(bearing_hht_data))
                    bearings_heatlh_states_tensor.append(torch.LongTensor(health_state))
            
            # Creating datasets.
            bearings_deep_feature_dataset = []
            for bearing_hht_tensor, bearing_heatlh_states_tensor in zip(bearings_hht_tensor, bearings_heatlh_states_tensor):
                bearings_deep_feature_dataset.append(self.ModelsDataset(bearing_hht_tensor, bearing_heatlh_states_tensor))

            # Creating data loaders.
            bearings_deep_feature_loader = []
            for bearing_deep_feature_dataset in bearings_deep_feature_dataset:
                bearings_deep_feature_loader.append(
                DataLoader(
                    bearing_deep_feature_dataset,
                    batch_size=models_params['cnn_batch_size'],
                    shuffle=False,
                    num_workers=4,
                ))

            # Saving CNN deep features loaders.
            with open(deep_features_loader_path, 'wb') as file:
                pickle.dump(bearings_deep_feature_loader, file)

            return bearings_deep_feature_loader

    def cnn_train(self, data_processed, models_params, predictor_params, device): 
        
        cnn_path = os.getcwd() + '/predictor/data/Processed_Data/cnn_model/cnn_model'

        # Loading saved model and hht_data pre-processed.
        if os.path.isfile(cnn_path):
            self.data_driven_models['cnn'].load_state_dict(torch.load(cnn_path))
            if predictor_params['return_cnn_model']:
                return
            
        # Getting data loaders.
        train_loader, eval_loader = self.cnn_data(data_processed, models_params, predictor_params, 'trainning')

        # Passing model to device and setting criterion.
        cnn_model = self.data_driven_models['cnn']
        cnn_model = cnn_model.to(device)

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

                # Forwarding data and getting prediction.
                train_output_ = cnn_model(train_feature)[0]

                # Computing loss and ajusting CNN weights.
                loss = criterion(train_output_, train_output)
                
                # Setting the parameter gradients to zero.
                optimizer.zero_grad()

                # Ajusting the weights of CNN.
                loss.backward()
                optimizer.step()
                
                # Averaging running_loss.
                running_loss = (running_loss*i + loss.item())/(i+1)

                # Calculating accuracy.
                wrong_predictions = 0
                for output_class, output_class_ in zip(train_output, train_output_):
                    if output_class.item() == 1:
                        if output_class_[1].item() < output_class_[0].item():
                            wrong_predictions += 1
                    elif output_class.item() == 0:
                        if output_class_[0]. item() < output_class_[1].item():
                            wrong_predictions += 1

            """ Try to calculate kappa score here. """

            if epoch % 1 == 0:
                print('Epoch', epoch)
                print('Loss Train %.4f' % running_loss)
                train_accuracy = (i - wrong_predictions) / i
                print('Train accuracy %.4f' % train_accuracy)

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

                wrong_predictions = 0
                for output_class, output_class_ in zip(eval_output, eval_output_):
                    if output_class.item() == 1:
                        if output_class_[1].item() < output_class_[0].item():
                            wrong_predictions += 1
                    elif output_class.item() == 0:
                        if output_class_[0]. item() < output_class_[1].item():
                            wrong_predictions += 1
                        
            if epoch % 1 == 0:
                end = time.time()
                print('\nLoss Test %.4f' % running_loss)
                test_accuracy = (i - wrong_predictions) / i
                print('Test accuracy %.4f' % test_accuracy)
                print('Time %.2f' % (end - start), 's\n')

        # Saving model.
        torch.save(cnn_model.state_dict(), cnn_path)

    def lstm_data(self, data_processed, models_params, predictor_params):
        
        deep_features = np.array(data_processed['deep_features'])
        rul = data_processed['rul']

        # Selecting deep features from fast degradation data.
        for i, (_, bearing_health_assessment) in enumerate(data_processed['health_assessment'].items()):
            deep_features[i] = deep_features[i][bearing_health_assessment['health_states']['fast_degradation'][0] : bearing_health_assessment['health_states']['fast_degradation'][1] + 1]

        """
        This part makes len(deep_features[i]) == len(rul[i])
        This is done because of 2 situations.
        1st - Bearing RMS pass the stop rms threshold.
        2nd - Bearing RMS don't pass the stop rms threshold (RMS data fitted in a polynomial to get RUL).
        
        For 1st, there are more deep features than RUL values because the we have more data after RUL = 0.
        For 2nd, there are less deep fetaures than RUL values because the test was stopped before the stop RUL threshold.
        """
        
        # Building rul array.
        recording_step_time = predictor_params['recording_step_time']
        for i, bearing_rul in enumerate(rul):
            rul[i] = np.arange(bearing_rul, -recording_step_time, step=-recording_step_time)

        # Slicing both to have the same length.
        for i, (bearing_rul, bearing_deep_features) in enumerate(zip(rul, deep_features)):
            min_len = min(len(bearing_rul), len(bearing_deep_features))
            rul[i] = rul[i][0:min_len]; deep_features[i] = deep_features[i][0:min_len]
        
        # Scaling deep_features.
        shapes = [bearing_deep_features.shape[0] for bearing_deep_features in deep_features]
        deep_features = np.concatenate(deep_features)
        deep_features = self.normalize_data(deep_features)

        # Restoring deep features shape.
        aux_shape = 0; deep_features_aux = []
        for shape in shapes:
            deep_features_aux.append(deep_features[aux_shape:aux_shape + shape])
            aux_shape = shape
        deep_features = deep_features_aux

        # Splitting train and eval data.
        train_deep_features = deep_features[1:]; eval_deep_features = [deep_features[0]]
        train_rul = rul[1:]; eval_rul = [rul[0]]

        # Here we get data and create many moving windows of size window_size to pass to LSTM.
        # LSTM is going to recieve a window and predict the RUL value.
        window_size = models_params['lstm_window_size']
        train_deep_features_window = []
        for i, (bearing_deep_features, bearing_rul) in enumerate(zip(train_deep_features, train_rul)):
            train_rul[i] = train_rul[i][window_size - 1:]
            bearing_deep_features_window = []
            for j in range(len(bearing_deep_features)):
                if j + window_size == len(bearing_deep_features):
                    break
                bearing_deep_features_window.append(bearing_deep_features[j : j + window_size])
            train_deep_features_window.append(bearing_deep_features_window)
        
        train_deep_features = train_deep_features_window

        # Creating tensors.
        if predictor_params['cuda_available']: 
            train_deep_features, eval_deep_features = torch.cuda.FloatTensor(train_deep_features), torch.cuda.FloatTensor(eval_deep_features)
            train_rul, eval_rul = torch.cuda.FloatTensor(train_rul), torch.cuda.FloatTensor(eval_rul)
        else:
            deep_feature_windows_tensor = []; rul_windows_tensor = []
            for window_deep_features, window_rul in zip(train_deep_features, train_rul):
                deep_feature_windows_tensor.append((torch.FloatTensor(window_deep_features)))
                rul_windows_tensor.append((torch.FloatTensor(window_rul)))
                
            train_deep_features, train_rul = deep_feature_windows_tensor, rul_windows_tensor
            eval_deep_features, eval_rul = torch.FloatTensor(eval_deep_features), torch.FloatTensor(eval_rul)

        return train_deep_features, train_rul, eval_deep_features, eval_rul

    def lstm_train(self, data_processed, models_params, predictor_params, device):
        
        lstm_path = os.getcwd() + '/predictor/data/Processed_Data/lstm_model/lstm_model'
        train_deep_features, train_rul, eval_deep_features, eval_rul = self.lstm_data(data_processed, models_params, predictor_params)

        #if os.path.isfile(lstm_path):
        #    self.data_driven_models['lstm'].load_state_dict(torch.load(lstm_path))

        # Passing model to device and setting criterion.
        lstm_model = self.data_driven_models['lstm']
        lstm_model = lstm_model.to(device)

        criterion = nn.MSELoss()

        for epoch in range(models_params['lstm_epochs']):
            optimizer = optim.SGD(lstm_model.parameters(), lr=0.001)

            # Setting model to trainning mode.
            lstm_model.train()      
            epoch_loss = 0.0
            for j, (bearing_train_deep_features, bearing_train_rul) in enumerate(zip(train_deep_features, train_rul)):
                bearing_loss = 0.0
                for i, (train_features, train_output) in enumerate(zip(bearing_train_deep_features, bearing_train_rul)):
                    # Passing data to device.
                    train_features, train_output = train_features.to(device), train_output.to(device)

                    # Initializing hidden states.
                    lstm_model.init_hidden(models_params['lstm'])

                    # Forwarding data and getting prediction.
                    # Unsqueeze because we are using 1 batch. Input of shape (seq_len, batch, input_size). See https://pytorch.org/docs/stable/nn.html for more details.
                    train_features = train_features.unsqueeze(1)
                    
                    train_output_ = lstm_model(train_features)
                    
                    # Computing loss.
                    loss = criterion(train_output_, train_output)
                    
                    # Setting the parameter gradients to zero.
                    optimizer.zero_grad()

                    # Ajusting the weights of LSTM.
                    loss.backward()
                    optimizer.step()

                    # Averaging running_loss.
                    bearing_loss = (bearing_loss*i + loss.item())/(i+1)
                epoch_loss = (epoch_loss*j + bearing_loss)/(j+1)
                print(epoch_loss)
                
            # Setting model to evaluation mode.
            #lstm_model.eval()
            #running_loss = 0.0
            #for i, (eval_features, eval_output) in enumerate(eval_loader):
            #
            #    # Passing data to device.
            #    eval_features, eval_output = eval_features.to(device), eval_output.to(device)
            #
            #    # Forwarding data and getting prediction.
            #    eval_output_ = lstm_model(eval_features)
            #    loss = criterion(eval_output_, eval_output)
            #
            #    # Averaging running_loss.
            #    running_loss = (running_loss*i + loss.item())/(i+1)
            #            
            #if epoch % 1 == 0:
            #    end = time.time()
            #    print('Loss Test %.4f' % running_loss)
            #    print('Time %.2f' % (end - start), 's\n')

        # Saving model.
        torch.save(lstm_model.state_dict(), lstm_path)
                
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
