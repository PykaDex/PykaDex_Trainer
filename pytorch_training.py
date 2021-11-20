import os
import cv2
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from pathlib import *

print('running')

#Location of training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Loading Training Data
clean_data_path = PurePath("../PykaDex_Data/Data/numpy_files/Pokemon_Data_Colour_GenX_80.npy")
bgswap_data_path = PurePath("../PykaDex_Data/Data/numpy_files/Pokemon_Data_Colour_GenX_BGswap_80.npy")
augmented_data_path = PurePath("../PykaDex_Data/Data/numpy_files/Pokemon_Data_Colour_GenX_Augmented_80.npy")

training_data_clean = np.load(clean_data_path, allow_pickle = True)
training_data_bgswap = np.load(bgswap_data_path, allow_pickle = True)
training_data_augmented = np.load(augmented_data_path, allow_pickle = True)

#training_data = np.concatenate((training_data_clean, training_data_bgswap, training_data_augmented), axis=0)
training_data = training_data_clean

#Separating Data into training and testing data
IMG_SIZE = 80
X = torch.Tensor([i[0] for i in training_data]).view(-1,3,IMG_SIZE,IMG_SIZE)
X = X/255.0 #scaling pixels
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1 
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]


#VGG16 model from torch
net = torchvision.models.vgg16(pretrained=True)

for param in net.parameters():
    param.requires_grad = False

net.features[0] = nn.Conv2d(3,64,kernel_size=(3,3), stride=(1,1), padding=(1,1))
net.classifier[6] = nn.Linear(4096,3)

net.to(device)

#ptimizer and Loss function
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

#Training
BATCH_SIZE = 64
EPOCHS = 20

def train(net):
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
           
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 3, IMG_SIZE, IMG_SIZE)
            batch_y = train_y[i:i+BATCH_SIZE]

            net.zero_grad()

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()    # Does the update

        print(f"Epoch: {epoch + 1}. Loss: {loss}")

############################
# chose which model to train
############################


#Accuracy
def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1, 3, IMG_SIZE, IMG_SIZE))[0]
            predicted_class = torch.argmax(net_out)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct/total, 3))


#Calling Functions
train(net)
test(net)

#Saving model
model_file_name = "model_trial_pokemon_3_2.pth"
model_directory = "Trained_models"
new_path = PurePath(model_directory, model_file_name)

answer = input("Would you like to save this model? (y/n) ")

if answer == "y":
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        torch.save(net.state_dict(), new_path)
        print("=> Download complete . Good Bye!")
    if os.path.exists(model_directory):
        torch.save(net.state_dict(), new_path)
elif answer == "n":
    print("Ok! Bye!")
    quit()
else:
    print("Hmmm not sure what you wrote... soooo.... bye!")
    quit()

