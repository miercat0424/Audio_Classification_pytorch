import torch
from train import FeedForwardNet, download_mnist_datsets

class_mapping = [
    "0","1","2","3","4","5","6","7","8","9"
]

def predict(model, input, target, class_mapping):
    model.eval()                                                # -> every time evaluate // gradients only needs when we actually training
    with torch.no_grad():
        predictions = model(input)                              # -> pass the input to model and get back to predictions
        # These are Tensor objects with a specific shape
        # Tensor (1, 10)                                          -> [[0.1, 0.01 , ... , 0.6]] if sum all of them it will get 1 point cuz of softmax
        # (1,10) 1 = numberof samples that we are passing to the model
        #       10 = the number of classes that the model tries to predicts ( we have 10 classes digits so 10 )
        # so we will catch out highest number of predictions 
        predicted_index = predictions[0].argmax(0)              # -> associated with the highest value & we'll take it for axis zero 
        # mapping this prediected index to relative class
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted , expected



if __name__ == "__main__" :

    # load back the model 
    feed_forward_net = FeedForwardNet()                         # -> ready to load pytorch model
    state_dict       = torch.load("feedforwardnet.pth")         
    feed_forward_net.load_state_dict(state_dict)                # -> directly to pytorch  loading back

    # load MNIST validation dataset
    _, validation_data = download_mnist_datsets()

    # get a sample from the validation datasets for inference
    input, target = validation_data[0][0], validation_data[0][1] # ->

    # make an inference
    predicted, expected = predict(feed_forward_net, input, target, 
                                    class_mapping)              

    print(f"Predicted : {predicted}, expected : {expected}")                                    

