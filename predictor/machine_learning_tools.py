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
from mpl_toolkits.mplot3d import Axes3D

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

    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
            super(NeuralNetworks.LSTM, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.batch_size = batch_size
            self.num_layers = num_layers

            # Define the LSTM layer.
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

            # Define the output layer.
            self.linear = nn.Linear(self.hidden_dim, output_dim)

        def init_hidden(self):
            # This is what we'll initialise our hidden state as.
            return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                    torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

        def forward(self, input):
            # Forward pass through LSTM layer.
            # shape of lstm_out: [input_size, batch_size, hidden_dim].
            # shape of self.hidden: (a, b), where a and b both.
            # have shape (num_layers, batch_size, hidden_dim).
            lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
            
            # Only take the output from the final timestep.
            # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction.
            y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
            return y_pred.view(-1)

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
        
        #if(os.path.isfile(deep_features_path)):
        #    with open(deep_features_path, 'rb') as file:
        #        return pickle.load(file)

        cnn_model = self.data_driven_models['cnn']
        cnn_model.eval()
        bearings_deep_features = []

        bearings_deep_feature_loader, _ = self.cnn_data(data_processed, models_params, predictor_params, 'deep_feature_extraction')

        criterion = nn.CrossEntropyLoss()        
        for bearing_deep_feature_loader in bearings_deep_feature_loader:
            bearing_deep_features = []
            for i, (eval_feature, eval_output) in enumerate(bearing_deep_feature_loader):
                # Passing data to device.
                eval_feature, eval_output = eval_feature.to(device), eval_output.to(device)

                # Forwarding data and getting deep features. Appending numpy arrays because tensors uses more memory.
                bearing_deep_features.extend([x.detach().numpy() for x in cnn_model(eval_feature)[1]])
                eval_output_ = cnn_model(eval_feature)[0]
                loss = criterion(eval_output_, eval_output)


                if i % 25 == 0:
                    print(models_params['cnn_batch_size']*(i+1), 'deep feature extracted.')
            bearings_deep_features.append(bearing_deep_features)
        
        # Saving pre-processed hht_data.
        with open(deep_features_path, 'wb') as file:
            pickle.dump(bearings_deep_features, file)
        
        data_processed['deep_features'] = bearings_deep_features

        # Calculating PCA of deep features.
        scaler = StandardScaler()
        #import ipdb; ipdb.set_trace()
        for i, bearing_deep_features in enumerate(bearings_deep_features):
            bearings_deep_features[i] = scaler.fit_transform(bearing_deep_features)
        
        pca = PCA(n_components=3)
        N = len(bearing_deep_features)
        pca_components_normal = pca.fit_transform(bearings_deep_features[0][0:40])
        pca_components_degradation = pca.fit_transform(bearings_deep_features[0][2760:2800])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        import ipdb; ipdb.set_trace()
        ax.scatter(pca_components_normal[:, 0], pca_components_normal[:, 1], pca_components_normal[:, 2])
        ax.scatter(pca_components_degradation[:, 0], pca_components_degradation[:, 1], pca_components_degradation[:, 2])

        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()

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
            hht_data = [files_hht for bearing_data in hht_data for files_hht in bearing_data[1]] 
            
            # Normalizing data.
            #scaler = StandardScaler()
            scaler = MinMaxScaler(feature_range=[-1,1])

            for i, data in enumerate(hht_data):
                hht_data[i] = scaler.fit_transform(np.reshape(hht_data[i], [-1, 1]))
            
            hht_data = np.squeeze(hht_data)

            # Saving hht_data at this point for deep features.
            with open(deep_features_hht_data_path, 'wb') as file:
                pickle.dump([np.split(hht_data, index_slices[:-1]), np.split(health_states, index_slices[:-1])], file)

            # Splitting data into traning data and validation data.
            train_hht, eval_hht, train_health_states, eval_health_states = train_test_split(hht_data, health_states, test_size=0.3, random_state=14, shuffle=True)

            # Oversampling fast degradation class. Must be after data splitting. Otherwise, we could could end up with the same observation in both datasets.
            # Selecting fast degradation data.
            #train_hht, train_health_states = SMOTE().fit_resample(train_hht, train_health_states)
            train_hht, train_health_states = RandomOverSampler().fit_resample(train_hht, train_health_states)

            # Shuffling data.
            random_index = np.random.permutation(len(train_hht))
            train_hht = np.array(train_hht)[random_index]
            train_health_states = np.array(train_health_states)[random_index]
            
            train_hht = [np.reshape(x, predictor_params['hht_cnn_shape']) for x in train_hht]
            eval_hht = [np.reshape(x, predictor_params['hht_cnn_shape']) for x in eval_hht]

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
            
            # Creating data loaders.
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
            
            bearings_deep_feature_dataset = []
            for bearing_hht_tensor, bearing_heatlh_states_tensor in zip(bearings_hht_tensor, bearings_heatlh_states_tensor):
                bearings_deep_feature_dataset.append(self.ModelsDataset(bearing_hht_tensor, bearing_heatlh_states_tensor))

            bearings_deep_feature_loader = []
            for bearing_deep_feature_dataset in bearings_deep_feature_dataset:
                bearings_deep_feature_loader.append(
                DataLoader(
                    bearing_deep_feature_dataset,
                    batch_size=models_params['cnn_batch_size'],
                    shuffle=True,
                    num_workers=3,
                ))

            return bearings_deep_feature_loader, data[1]

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
                loss.backward(loss)
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

            # Try to calculate kappa score here.
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
                print('Time %.2f' % (end - start), 's')
                test_accuracy = (i - wrong_predictions) / i
                print('Test accuracy %.4f' % test_accuracy, '\n')

        # Saving model.
        torch.save(cnn_model.state_dict(), cnn_path)

    def lstm_train(self, data_processed, models_params, predictor_params, device):
        pass

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
