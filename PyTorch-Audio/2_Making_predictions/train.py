import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

# 1- download dataset
# 2- create data loader
# 3- build model
# 4- train
# 5- save trained model
# ---------------------------------------------------------------------------------

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001


# 3 build model---------------------------------------------------------------------------------
class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten      = nn.Flatten()                    # -> 
        self.dense_layers = nn.Sequential(                  # -> number of layers will contain more and more

        # Sequential will floor 1 ~ N layers 
            nn.Linear(28*28,256),                           # -> simple Dense Layer // images are 28*28 in MNIST , 256 Neorun
            nn.ReLU(),
            nn.Linear(256,10),                              # -> this is the labels in MNIST
        )    
        self.softmax = nn.Softmax(dim=1)                    # -> Basic transformation , from different 10 classes and if you sum all the things it will be 1.

    def forward(self,input_data):                           # -> indicate PyTorch how to manipulate the dating (Squential)

        flattened_data = self.flatten(input_data)           # -> passing input_data and will be comeout to flatten vali
        logits = self.dense_layers(flattened_data)          # -> passing flattened_data and the dense_layers will comeout logits vali
        predictions = self.softmax(logits)

        return predictions


# 1 download datasets---------------------------------------------------------------------------------
def download_mnist_datsets():
    train_data = datasets.MNIST(        # -> datasets class // it allows to store label , data , use and training
        root="data",                    # -> where to store datasets
        download=True,                  # -> if i don't have datasets then download plz
        train=True,                     # -> in train sets
        transform=ToTensor()            # -> apply sort of transformation of datasets
                                        # -> reshape tensor to 0 or 1
    )   
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,                    # ->    
        transform=ToTensor()
    )
    return train_data , validation_data

# 4 train---------------------------------------------------------------------------------
def train_one_epoch(model, data_loader, loss_fn, optimiser, device):                  # -> what training model
    # loop for all samples  // batch access and wide
    for inputs, targets in data_loader:                                               # -> sets cal , backpropagate loss , weight each batchs
        inputs , targets = inputs.to(device) , targets.to(device)                     

        # calculate loss
        predictions = model(inputs)                                 # -> need to get prediction
        loss        = loss_fn(predictions,targets)                  # -> expected values and prediction

        # backpropagate loss and update weights 
        optimiser.zero_grad()                                       # -> cal gradient to decide how to update // each iteration gets to 0 gradients
        loss.backward()
        optimiser.step()                                            # -> updating weights // optimiser is the important thing on weighting
        
    print(f"Loss : {loss.item()}")                                 # -> printing the loss for last batch that have


def train(model, data_loader, loss_fn, optimiser, device, epochs):         # -> each iteration 
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------------------------")
    print("Training is Done.")

    pass



if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datsets()
    print("MNIST dataset downloaded")

    # 2 create data loader---------------------------------------------------------------------------------
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # 3 builds model (excute)---------------------------------------------------------------------------------
    if torch.cuda.is_available():                       # check CPU or CUDA
        device = "cuda"
    else : 
        device = "cpu"
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)      # What will you using CUDA or CPU

    # 4 instantiate loss function + optimiser ---------------------------------------------------------------------------------
    loss_fn     = nn.CrossEntropyLoss()
    optimiser   = torch.optim.Adam(feed_forward_net.parameters(),     # -> call all of the parameters
                                   lr = LEARNING_RATE)


    # 4 train model ---------------------------------------------------------------------------------
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    # 5 Save trained model ---------------------------------------------------------------------------------
    torch.save(feed_forward_net.state_dict(),"feedforwardnet.pth") # -> python dictionary has state which has informations // stroring the file name .pth file
    print("Model trained and sotred at feedforwardnet.pth")