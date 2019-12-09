"""
@author: Chesney Carolissen
@title: Image Classifier training file
"""

# ------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------- #

import argparse # The argparse module makes it easy to write user friendly command-line interfaces.

# Machine learning toolkits
import torch
from torch import nn
from torch import optim # The optimizer functions (eg. Adam)
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os.path

# ------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------- #

"""
Define the parser function which parses keywords from the command line.
"""
def fn_parser():
    parser = argparse.ArgumentParser(
        description="Image Classifier Neural Network")
    
    # DONE : Add Architecture selection to parser
    parser.add_argument('--arch', type=str,
                        help='Choose the architecture from the torchvision.models as a string')
    
    # DONE : Add Checkpoint directory to parser
    parser.add_argument('--save_dir', type=str,
                       help='Set the save directory for checkpoints.If undefined, the model will be lost.')
    
    # DONE : Add the learning rate
    parser.add_argument('--learning_rate', type=float,
                        help='The gradient descent learning rate as float.')
    
    # DONE : Add the hidden units
    parser.add_argument('--hidden_units', type=int,
                        help='Hidden units for the Dense Neural Network classifier.')
    
    # DONE : Add the epochs
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs for training as int.')
    
    # DONE : Add the GPU option to parser  
    parser.add_argument('--gpu', action="store_true",
                        default=False,
                        help='Enable GPU and CUDA for calculations')
    # DONE : Parse the arguments
    args = parser.parse_args()
    return args

"""
The transformer function. Performs training on transformations on a dataset.
"""
def fn_train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

"""
Test the transformer. Performs validation transformation on a dataset.
"""
def fn_test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

"""
This function creates a dataloader from the imported dataset.
"""
def fn_data_loader(data, train=True):
    
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    
    return loader

"""
Check if the user enabled GPU support. CUDA with GPU or CPU
"""
def fn_activate_gpu(gpu_arg):
    
    # Return the CPU device if the GPU is False
    if not gpu_arg:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if device == "cpu":
        print("The CUDA device was not found. Using the CPU instead.")
    
    return device

"""
Primary loader model
"""
def fn_primary_loader_model(architecture="vgg16"):
    
    if type(architecture) == type(None):
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print(f"Network architecture set as {model.name}")
    else:
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    # We need to freeze the parameters so that we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
    return model

"""
Initial Classifier :
Create a classifier with the correct number of input layers.
"""
def fn_initial_classifier(model, hidden_units):
    
    if type(hidden_units) == type(None): 
        hidden_units = 4096
        print(f"Number of Hidden Layers specified as {hidden_units}.")
    
    input_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

"""
Validation Function
Validates the training against the testloader to return loss and accuracy.
"""
def fn_validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for il, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

"""
Function network represents the training of the network model.
"""

def fn_network_trainer(Model, Trainloader, Testloader, ValidLoader,
                    Device, Criterion, Optimizer, 
                    Epochs, Print_every, Steps):
    
    if type(Epochs) == type(None):
        Epochs = 5
        print(f"Number of Epochs {Epochs}.")    
 
    print("The training process is initializing, please wait...")
    
    # Now we train the model.
    for e in range(Epochs):
        running_loss = 0
        Model.train()
        
        # Here we do the forward pass and back prop.
        for ij, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            inputs, labels = inputs.to(Device), labels.to(Device)
            Optimizer.zero_grad()
            outputs = Model.forward(inputs)
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
            running_loss += loss.item()
        
            if Steps % Print_every == 0:
                Model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = fn_validation(Model, ValidLoader, Criterion, Device)   
                print(f'Epoch: {(e+1)} / {Epochs} | '
                      f'Training Loss: {(running_loss / Print_every):.4f} | '
                      f'Validation Loss: {(valid_loss / len(Testloader)):.4f} | '
                      f'Validation Accuracy: {(accuracy / len(Testloader)):.4f}')
                running_loss = 0
                Model.train()

    return Model

#Function validate_model(Model, Testloader, Device) validate the above model on test data images
def fn_validate_model(Model, Testloader, Device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

"""
Save the model at a defined checkpoint
"""
def fn_initial_checkpoint(Model, save_dir, Train_data):
    
    if type(save_dir) == type(None):
        print(f"Model check point directory not found {save_dir}.The model will not be saved")
    else:
        if os.path.isdir(save_dir):
            Model.class_to_idx = Train_data.class_to_idx
            checkpoint = {
                'architecture' : Model.name,
                'classifier': Model.classifier,
                'class_to_idx': Model.class_to_idx,
                'state_dict': Model.state_dict()
            }
            save_path = os.path.join(save_dir, 'checkpoint.pth')
            torch.save(checkpoint, save_path)
        else:
            print(f"Directory: {save_dir} Not Found. Model will not be saved")
            
            
        
# ------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------- #
def main():
    
    # DONE : Check the arguments that were passed in 
    # TODO : Wrap things in try/catch blocks
    
    # Get the keyword arguments for training
    args = fn_parser()
    
    # Setup the directories for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass the transforms then create the trainloader
    train_data = fn_test_transformer(train_dir)
    valid_data = fn_test_transformer(valid_dir)
    test_data = fn_test_transformer(test_dir)
    
    # Training loaders
    train_loader = fn_data_loader(train_data)
    valid_loader = fn_data_loader(valid_data, train=False)
    test_loader = fn_data_loader(test_data, train=False)
    
    model = fn_primary_loader_model(architecture=args.arch)
    
    model.classifier = fn_initial_classifier(model, hidden_units=args.hidden_units)
    
    # Check if the user activated GPU support
    device = fn_activate_gpu(gpu_arg=args.gpu)
    
    model.to(device)
    
    # Check for the learn rate arguments
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print(f'Learning rate specified as {learning_rate}')
    else:
        learning_rate = args.learning_rate
        
    # Define the loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Let's define the deep learning method
    print_every = 30
    steps = 0
    
    # Training the classifier layers using back propogation
    trained_model = fn_network_trainer(model, train_loader, valid_loader, test_loader,
                                   device, criterion, optimizer, args.epochs,
                                   print_every, steps)
    
    print("The training process is now complete.")
    
    # Validate the Model
    fn_validate_model(trained_model, test_loader, device)
    
    # Save the model
    fn_initial_checkpoint(trained_model, args.save_dir, train_data)
    
if __name__ == "__main__": main()
