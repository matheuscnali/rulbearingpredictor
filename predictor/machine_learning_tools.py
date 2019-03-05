import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision    
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config import CONF


class NeuralNetworks():
    
    data_driven_models = {}

    def __init__(self):
        pass

    # Dataset class for Pytorch.
    class PredictorDataset(Dataset):
        def __init__(self, data, target, transform=None):
            self.data = torch.FloatTensor(data)
            self.target = torch.FloatTensor(target)
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
            self.fc1 = nn.Linear(*params['linear1'])
            self.fc2 = nn.Linear(*params['linear2'])

        def forward(self, x):
            in_size = x.size(0)
            
            x = F.relu(self.pool(self.conv1(x)))
            x = F.relu(self.pool(self.conv2(x)))
            
            # Flatten the tensor.
            x = x.view(in_size, -1)
            x = F.softmax(self.fc1(x), dim=1)
            x = F.softmax(self.fc2(x), dim=1)

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

    def predict(self, data_processed, models_params, predictor_params):
        
        # Setting to use GPU processor if avaiable.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for current_bearing in predictor_params['bearings']:
            current_bearing = str(current_bearing)

            # Convolutional Neural Network (CNN)
            # Transforming health state data to one hot vector.
            health_states_hot_vector = []
            health_states = data_processed['health_assessment'][current_bearing]['health_states']
            
            for key, health_state in health_states.items():
                if key == 'normal':
                    normal_qty = health_state[1] - health_state[0] + 1
                    health_states_hot_vector = [[1,0]]*normal_qty
                elif key == 'fast_degradation':
                    fast_degradation_qty = health_state[1] - health_state[0] + 1
                    health_states_hot_vector.extend([[0,1]]*fast_degradation_qty)

            # Normalizing HHT data.
            hht_data = [x[1] for x in data_processed['hht_marginal_spectrum'][current_bearing]]

            hht_data_normalized = []
            for data in hht_data:
                hht_data_normalized.append(self.normalize_data(data))

            # Reshaping HHT data and splitting data into traning data and validation data.
            train_percentage = 0.7; N = len(hht_data); M = len(health_states_hot_vector)
            # shape = [batch_size, color_channels, height, width]. In this case we set a number of batch files to the number of files which we will process per epoch. Height and width are user defined. 'color_channels' would be the depth which in our case is just 1.
            hht_data = [np.reshape(x, [1, 32, 40]) for x in hht_data]

            train_hht, eval_hht = np.split(hht_data, [int(train_percentage*N)])
            train_health_states, eval_health_states = np.split(health_states_hot_vector, [int(train_percentage*M)])
            
            # Creating dataset.
            train_dataset = self.PredictorDataset(train_hht, train_health_states)
            eval_dataset = self.PredictorDataset(eval_hht, eval_health_states)
           
            # Creating loaders.
            train_loader = DataLoader(
                train_dataset,
                batch_size=models_params['cnn_batch_size'],
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            eval_loader = DataLoader(
                eval_dataset,
                batch_size=models_params['cnn_batch_size'],
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            # Passing model to device and setting optimizer and criterion.
            model = self.data_driven_models['cnn'].to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(models_params['cnn_epochs']):
                # Setting model to trainning mode.
                model.train()

                for batch_id, (hht_data, health_states) in enumerate(train_loader):
                    
                    # Passing data to device.
                    # VERIFICAR PRQ TÀ COM O HEALTH STATES NA ORDEM ERRADA.
                    hht_data, health_state = hht_data.to(device), health_states.to(device)
                    
                    # Setting the parameter gradients to zero.
                    optimizer.zero_grad()

                    # Forwarding data and getting prediction.
                    health_states_ = model(hht_data)

                    # Computing loss and ajusting CNN weights.
                    loss = criterion(health_states_, health_states)
                    loss.backward(loss)
                    
                    optimizer.step()

                if epoch % 5 == 0:    
                    print('Epoch', epoch)
                    print('Loss', loss.data.item())

    def adjust_learning_rate(self, epoch):
        lr = 0.001

        if epoch > 180:
            lr = lr / 1000000
        elif epoch > 150:
            lr = lr / 100000
        elif epoch > 120:
            lr = lr / 10000
        elif epoch > 90:
            lr = lr / 1000
        elif epoch > 60:
            lr = lr / 100
        elif epoch > 30:
            lr = lr / 10






