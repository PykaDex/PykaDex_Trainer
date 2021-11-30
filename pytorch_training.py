import os
import cv2
import time 
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from pathlib import *

print('Running...')

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
def fwd_pass(X, y, train=False):

    if train:
        net.zero_grad()
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss

def test(size):
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1,3,IMG_SIZE,IMG_SIZE).to(device), y.to(device))
    return val_acc, val_loss

MODEL_NAME = f"model-{int(time.time())}"  # gives a dynamic model name, to just help with things getting messy over time. 

BATCH_SIZE = 100
EPOCHS = 3
def train(net):

    with open("model.log", "a") as f: #creates a log file with all the accuracies and losses saved
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,3,IMG_SIZE,IMG_SIZE)
                batch_y = train_y[i:i+BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                if i % 32 == 0:
                    val_acc, val_loss = test(BATCH_SIZE)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")

############################
# chose which model to train
############################


#Calling Functions
train(net)

#Saving model
model_file_name = f"Model_pokemon_{IMG_SIZE}_{EPOCHS}.pth"
model_directory = "Trained_models"
new_path = os.path.join(model_directory, model_file_name)

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

