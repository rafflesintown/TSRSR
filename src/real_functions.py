import numpy as np
import scipy.io 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler



def load_Boston():
    file = "data/boston_data.mat"
    data = scipy.io.loadmat(file)
    print("data.keys", data.keys())
    print(data['x'][:,0])


def load_bCancer():
    file = "data/cancer_data.mat"
    data = scipy.io.loadmat(file)
    print("data.keys", data.keys())
    print(data['t'])


class one_hidden_net(nn.Module):
    def __init__(self, n_feat = None, neurons = None):
        super(one_hidden_net, self).__init__()
        self.fc1 = nn.Linear(n_feat,neurons)
        self.fc2 = nn.Linear(neurons,1)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def RMSELoss(yhat,y):
    return torch.sqrt(F.mse_loss(yhat,y))



# class bCancer:
#     def __init__(self, file = "/Users/zhr568/desktop/research/batch_bayesian/dbo/src/data/cancer_data.mat"):
#         torch.manual_seed(10)
#         self.domain = np.array([[0, 10], [1e-3, 0.1], [0.1,0.9],[0.0,0.5]])
#         self.min = 0
#         self.arg_min = np.array([[1, 0.1,0.1,0.0]]) #fake argmin!
#         self.X = scipy.io.loadmat(file)['x']
#         self.Y = scipy.io.loadmat(file)['t']
#         # print("self.X shape", self.X.shape)
#         # print("self.Y shape", self.Y.shape)
#         self.X = np.transpose(self.X)
#         self.X = StandardScaler().fit_transform(X = self.X) #standardize
#         self.Y = np.transpose(self.Y)
#         self.Y = self.Y[:,0] #binary prediction, omit the second column
#         self.inputs = torch.tensor(self.X, dtype = torch.float32)
#         self.targets = torch.tensor(self.Y, dtype = torch.float32)
#         self.train_tensor_dataset = TensorDataset(self.inputs,self.targets)
#         self.val_size= int(0.2 * len(self.train_tensor_dataset))
#         train_size = len(self.train_tensor_dataset) - int(0.2 * len(self.train_tensor_dataset))
#         self.batch_size = 50
#         self.train_data, self.val_data = random_split(self.train_tensor_dataset, [train_size, self.val_size])
#         # print("self.X shape", self.X.shape)
#         # print("self.Y shape", self.Y.shape)
#         self.epochs = 10
#         self.n_feat = 9 #there are 13 features (excluding price) in Boston Housing dataset
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.function = lambda x: self.loss(x)


#     def loss(self, x):
#         # self.neurons = int(10 *x[0])
#         self.neurons = int(x[0])
#         self.lr = x[1]
#         self.weight_decay = x[2]
#         self.momentum = x[3]
#         # self.rho = x[3]
#         loss = self.train().detach().numpy()
#         # print("this is loss", loss)
#         return(loss)

#     def train(self):
#         model = one_hidden_net(n_feat = self.n_feat, neurons = self.neurons)
#         model.to(self.device)
#         loss_fn = self.binaryLoss
#         # optimizer = torch.optim.Adam(model.parameters(), lr = self.lr,
#         #     weight_decay = self.lr_decay)
#         optimizer = torch.optim.RMSprop(model.parameters(), lr = self.lr,
#             weight_decay = self.weight_decay, momentum = self.momentum)
#         # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
#         num_epochs = self.epochs
#         train_dataloader = DataLoader(self.train_data, self.batch_size, shuffle=True)
#         val_dataloader = DataLoader(self.val_data, self.val_size)

#         # # Train the model.
#         val_loss = self.fit(num_epochs, model, loss_fn, optimizer, train_dataloader, val_dataloader)
#         return(val_loss)

#     def binaryLoss(self,yhat,y):
#         loss = torch.nn.BCEWithLogitsLoss()
#         return loss(yhat[:,0], y)


#     def val_loss(self,y_out, y):

#         val_loss = torch.mean((torch.abs(y-y_out) < 0.5).float())
#         return val_loss


#     def fit(self,num_epochs, model, loss_fn, optimizer, train_dataloader, val_loader):
#         best_loss = 1e3
#         for epoch in range(num_epochs):
#             for inputs, targets in train_dataloader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 # Get predictions.
#                 preds = model(inputs)
                
#                 # Get loss.
#                 loss = loss_fn(preds, targets)
                
                
#                 # Compute gradients.
#                 loss.backward()
#     #             print(loss.item())
                
#                 # Update model parameters i.e. backpropagation.
#                 optimizer.step()
                
#                 # Reset gradients to zero before the next epoch.
#                 optimizer.zero_grad()
                
#             if (epoch + 1) % 50 == 0 or epoch ==0:
#                 # Get validation loss as well.
#                 for val_input, val_targets in val_loader:
#                     val_input, val_targets = val_input.to(self.device), val_targets.to(self.device)
#                     out = model(val_input)
#                     val_loss = self.val_loss(out, val_targets)
#                     if val_loss < best_loss:
#                         best_loss = val_loss
#             # print("Epoch [{}/{}], Training loss: {:.4f}, Validation Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item(), val_loss)) # Report loss value after each epoch.
#         return best_loss


class Boston:
    def __init__(self, file = "/Users/zhr568/desktop/research/batch_bayesian/dbo/src/data/boston_data.mat"):
        torch.manual_seed(10)
        self.domain = np.array([[1, 10], [1e-3, 0.1], [0.1,0.9],[0.0,0.5]])
        self.min = 0
        self.arg_min = np.array([[1, 0.1,0.1,0.0]]) #fake argmin!
        self.X = scipy.io.loadmat(file)['x']
        self.Y = scipy.io.loadmat(file)['t']
        self.X = np.transpose(self.X)
        self.X = StandardScaler().fit_transform(X = self.X) #standardize
        self.Y = np.transpose(self.Y)
        self.inputs = torch.tensor(self.X, dtype = torch.float32)
        self.targets = torch.tensor(self.Y, dtype = torch.float32)
        self.train_tensor_dataset = TensorDataset(self.inputs,self.targets)
        self.val_size= int(0.1 * len(self.train_tensor_dataset))
        train_size = len(self.train_tensor_dataset) - int(0.1 * len(self.train_tensor_dataset))
        self.batch_size = 50
        self.train_data, self.val_data = random_split(self.train_tensor_dataset, [train_size, self.val_size])
        # print("self.X shape", self.X.shape)
        # print("self.Y shape", self.Y.shape)
        self.epochs = 10
        self.n_feat = 13 #there are 13 features (excluding price) in Boston Housing dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.function = lambda x: self.loss(x)


    def loss(self, x):
        # self.neurons = int(5 *x[0])
        self.neurons = int(10 * x[0])
        self.lr = x[1]
        self.weight_decay = x[2]
        self.momentum = x[3]
        # self.rho = x[3]
        loss = self.train().detach().numpy()
        # print("this is loss", loss)
        return(loss)

    def train(self):
        model = one_hidden_net(n_feat = self.n_feat, neurons = self.neurons)
        model.to(self.device)
        loss_fn = self.RMSELoss
        # optimizer = torch.optim.Adam(model.parameters(), lr = self.lr,
        #     weight_decay = self.lr_decay)
        optimizer = torch.optim.RMSprop(model.parameters(), lr = self.lr,
            weight_decay = self.weight_decay, momentum = self.momentum)
        # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
        num_epochs = self.epochs
        train_dataloader = DataLoader(self.train_data, self.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_data, self.val_size)

        # # Train the model.
        val_loss = self.fit(num_epochs, model, loss_fn, optimizer, train_dataloader, val_dataloader)
        return(val_loss)

    def RMSELoss(self,yhat,y):
        return torch.sqrt(F.mse_loss(yhat,y))

    def fit(self,num_epochs, model, loss_fn, optimizer, train_dataloader, val_loader):
        best_loss = 1e3
        for epoch in range(num_epochs):
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Get predictions.
                preds = model(inputs)
                
                # Get loss.
                loss = loss_fn(preds, targets)
                
                
                # Compute gradients.
                loss.backward()
    #             print(loss.item())
                
                # Update model parameters i.e. backpropagation.
                optimizer.step()
                
                # Reset gradients to zero before the next epoch.
                optimizer.zero_grad()
                
            if (epoch + 1) % 50 == 0 or epoch ==0:
                # Get validation loss as well.
                for val_input, val_targets in val_loader:
                    val_input, val_targets = val_input.to(self.device), val_targets.to(self.device)
                    out = model(val_input)
                    val_loss = self.RMSELoss(out, val_targets)
                    if val_loss < best_loss:
                        best_loss = val_loss
                # print("Epoch [{}/{}], Training loss: {:.4f}, Validation Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item(), val_loss)) # Report loss value after each epoch.
        return best_loss




if __name__ == '__main__':
    # branin = Branin()
    # fun = lambda x: branin.function(x)
    # print(fun(branin.arg_min[1]))
    # boston = Boston()
    load_bCancer()
    bCancer = bCancer()
    x = [2,0.01,0.0, 0.8]
    loss = bCancer.loss(x)
    # print("loss", loss)


