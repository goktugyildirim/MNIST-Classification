#!/usr/bin/env python
# coding: utf-8

# ### Read Data

# In[2]:


import pandas as pd
import numpy as np
import collections, numpy

train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

x_train = np.array(train_data)[:,1:785]
y_train = np.array(train_data)[:,0].reshape((60000,1))

x_validation = x_train[:10000]
y_validation = y_train[:10000]

x_train = x_train[10000:60000]
y_train = y_train[10000:60000]

x_test = np.array(test_data)[:,1:785]
y_test = np.array(test_data)[:,0].reshape((10000,1))


# ### Show an image

# In[4]:


import matplotlib.pyplot as plt
plt.imshow(x_train[30].reshape(28,28))


# ### Distribution of Classes

# In[5]:


import matplotlib.pyplot as plt
 
plt.style.use('ggplot')
plt.hist(y_train[:10000], bins=30)
plt.title("Training Set")
plt.xlabel("Classes")
plt.ylabel("Class Counts")
plt.show()

plt.figure()

plt.style.use('ggplot')
plt.hist(y_validation[:2000], bins=30)
plt.title("Validation Set")
plt.xlabel("Classes")
plt.ylabel("Class Counts")
plt.show()

plt.figure()

plt.style.use('ggplot')
plt.hist(y_test[:2000], bins=30)
plt.title("Test Set")
plt.xlabel("Classes")
plt.ylabel("Class Counts")
plt.show()


# ### Data Normalizing

# In[6]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_normalized = scaler.fit_transform(x_train)
x_validation_normalized = scaler.fit_transform(x_validation)
x_test_normalized = scaler.fit_transform(x_test)
print(x_train_normalized.shape,x_validation_normalized.shape,x_test_normalized.shape)


# ### Make 4D Tensor for CNN

# In[7]:


x_train_normalized = x_train_normalized.reshape((50000,1,28,28))
x_validation_normalized = x_validation_normalized.reshape((10000,1,28,28))
x_test_normalized = x_test_normalized.reshape((10000,1,28,28))
print(x_train_normalized.shape,x_validation_normalized.shape,x_test_normalized.shape)


# In[8]:


y_train[0]


# ### One Hot Encoded Labels

# In[9]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
y_train = enc.fit_transform(y_train).toarray()
y_validation = enc.fit_transform(y_validation).toarray()
y_test = enc.fit_transform(y_test).toarray()
print(y_train.shape,y_validation.shape,y_test.shape)


# In[10]:


y_train[0]


# ### Make PyTorch Tensor

# In[11]:


import torch
x_train_normalized = torch.tensor(x_train_normalized).float()
x_validation_normalized = torch.tensor(x_validation_normalized).float()
x_test_normalized = torch.tensor(x_test_normalized).float()
y_train = torch.tensor(y_train).float()
y_validation= torch.tensor(y_validation).float()
y_test = torch.tensor(y_test).float()


# In[12]:


y_test[0]


# ### Model

# In[19]:


from IPython.core.debugger import set_trace # debug
import torch.nn as nn
import torch.optim as optimizer
#-----------------------------------
#Settings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import cv2
import sys
import numpy
from sklearn.model_selection import train_test_split
import pandas as pd #data processing
import warnings
import matplotlib.image as mpimg
import torch
warnings.filterwarnings('ignore')
numpy.set_printoptions(threshold=sys.maxsize) #full print setting
from sklearn.datasets import make_regression


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        channel = 50
        self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = channel, kernel_size = 5) 
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(in_channels = channel, out_channels =channel, kernel_size = 1)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        
        self.dropout1 = nn.Dropout2d(0.25)
        
        #self.dropout2 = nn.Dropout2d(0.5)
        
        self.pool = nn.MaxPool2d(2, 2)
    
        self.beta1 = nn.Linear(channel*2*2, 3)
        #torch.nn.init.xavier_uniform(self.beta1.weight)
        self.beta2 = nn.Linear(3, 10)
        #torch.nn.init.xavier_uniform(self.beta2.weight)
        self.beta3 = nn.Linear(10, 30)
        self.beta4 = nn.Linear(30, 10)
        #torch.nn.init.xavier_uniform(self.beta3.weight)
        
        self.output_of_CNN = []
        self.output_of_ANN = []
        
    def forward(self, X, training, i):
        
        channel=50
       
        y=self.conv1(X) 
        y=torch.nn.functional.relu(y)
        y=self.pool(y) 
        
        y=self.conv2(y) 
        y=torch.nn.functional.relu(y)
        y=self.pool(y) 
        
        y=self.conv3(y) 
        y=torch.nn.functional.relu(y)
        y=self.pool(y)
        
        
        
        
        
    
        y = y.view(X.shape[0], channel*2*2)
        
        if training == True and i==998:
            self.output_of_CNN.append(y)
            
        
        y=self.dropout1(y)
        
        
        #print(y.shape)
     
        y=self.beta1(y)
        
        #print("debug")
        
        y=torch.nn.functional.relu(y)
        y=self.beta2(y)
        y=torch.nn.functional.relu(y)
        
        y=self.dropout1(y)
        
        y=self.beta3(y)
        y=self.beta4(y)
    
        y = torch.nn.functional.softmax(y, dim=None)
        
        if training == True and i==998:
            outputs = [torch.argmax(sample).detach().numpy() for sample in list(y)]
            self.output_of_ANN.append(outputs)
        
        return y
    
    def predict(self, model, x_test, y_test):
        import numpy as np
        y_test = y_test.detach().numpy()
        
        model.eval() 
        with torch.no_grad():
            predictions = model(x_test, False,0)
            
        predictions = predictions.detach().numpy()
        one_hot_pred = [ 1 for i in range(predictions.shape[0]) if np.argmax(predictions[i]) == np.argmax(y_test[i]) ]
        accuracy = (len(one_hot_pred)/predictions.shape[0])*100
        return accuracy
        
#Parameters      
#***************************************************************************************************************************
learning_rate = 0.001
epoch = 1000
savingRate = 1 # Loading and making prediction at epoch every loadParameters times 
model = Net()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#***************************************************************************************************************************
epoch_list = []
validation_loss_list = []
training_loss_list = []
accuracy_list = []

for i in range(epoch):
    
        epoch_list.append(i)
        
        #Training
        model.train()
        optimizer.zero_grad() 
        y_prediction = model(x_train_normalized[:10000], True,i) 
        loss = loss_fn(y_prediction, y_train[:10000]) 
        training_loss_list.append(loss.item())
        loss.backward() 
        optimizer.step() 
        print("[Epoch {}] : Training Loss: {}".format(i,loss))
        
        
        #Evaluation
        model.eval() 
        with torch.no_grad():
            y_validation_prediction = model(x_validation_normalized[:2000],False,0)  
        loss = loss_fn(y_validation_prediction, y_validation[:2000])
        validation_loss_list.append(loss.item())
        print("          Validation Loss: {}".format(loss))
        
        
        #Performance
        accuracy = model.predict(model, x_test_normalized[:2000], y_test[:2000])
        accuracy_list.append(accuracy)
        print("          Accuracy in the Test Set: %{}".format(accuracy))
        
        
        #Save the Model
        if(i%savingRate==0): 
            #ModelSaving
            checkpoint = {'model': Net(),
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()}

            torch.save(checkpoint, 'models/Epoch {} Checkpoint.pth'.format(i))
            print("Model kaydedildi.")
        
            
    
        print("************************************************")
        


# In[ ]:





# In[28]:


fig, ax = plt.subplots()
ax.plot(epoch_list, training_loss_list, label="Training Loss")
ax.plot(epoch_list, validation_loss_list, label="Validation Loss")
ax.set_xlabel("epoch")
ax.set_title("MSE Loss ")
ax.legend();

fig, ax = plt.subplots()
ax.plot(epoch_list, accuracy_list)
ax.set_title("Accurcacy on the Test Set ")
ax.set_xlabel("epoch")
ax.set_ylabel("Percentage")
ax.legend();


# In[27]:


model


# In[ ]:


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

filepath = "Epoch 999 Checkpoint.pth"
model = load_checkpoint(filepath)


# ### I apply k-Means clustering to output of pre-trained CNNs

# In[62]:


cnn_output = model.output_of_CNN[0].detach().numpy()
ann_output = np.array(model.output_of_ANN[0]).reshape((10000,1)).ravel()
print("CNN output shape:", cnn_output.shape)
print("ANN output shape:", ann_output.shape)


# In[129]:


x_train = x_train_normalized[:100].detach().numpy().reshape((100,784))


# ### k-Means on initial x_train

# In[130]:


from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(x_train)
print(kmeans.labels_)


# ### k-Means on pre-trained CNNs

# In[131]:


from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=10, random_state=0)
z = kmeans.fit(cnn_output)


# In[132]:


print("Pre-trained CNN then k-Means clustering:\n", kmeans.labels_[:100]) # CNN + k-Means
print("Pre-trained CNN then ANN classification:\n",ann_output[:100]) #Accuracy is %91.35 => CNN + ANN


# In[ ]:




