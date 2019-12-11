"""
@author: Chesney Carolissen
@title: Image Classifier prediction file
@usage Run the below command to compare the image
python predict.py --image flowers/test/10/image_07090.jpg --checkpoint train_results/checkpoint.pth --top_k 3 --category_names cat_to_name.json
"""

# ------------------------------------------------------------------------------- #
# Libraries
# ------------------------------------------------------------------------------- #

import argparse
import numpy as np
import json
import PIL
import torch
from torchvision import models
import math

from train import fn_activate_gpu as check_gpu

# ------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------- #

def fn_parser():
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    parser.add_argument('--image', 
                        type=str, 
                        help='Point to the image file for prediction.',
                        required=True)
 
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose the top K matches as integer.')

    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to the checkpoint file.',
                        required=True)
   
    parser.add_argument('--category_names', 
                        type=str, 
                        help='The Mapping from Categories to real names.')

    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use CUDA with the GPU for calculations')

    args = parser.parse_args()
    
    return args

"""
Loads our saved deep learning model from a checkpoint.
"""
def fn_load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load more settings from the checkpoint
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

"""
Process the image by providing cropping, scaling and rotation
"""
def fn_process_image(image_path):
    test_image = PIL.Image.open(image_path)

    # Get the original dimensions of the test image.
    orig_width, orig_height = test_image.size

    # Here we find the shorter size
    # Create settings to crop to 256 (shortest side) 
    if orig_width < orig_height: 
        resize_size=[256, 256**600]
    else: 
        resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    # Find the pixels to crop on to create a 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Convert to numpy - 244x244 image with 3 channels (RGB)
    np_image = np.array(test_image)/255 # We divide by 255 because imshow() expects integers (0:1)

    # Now we normalize each color channel.
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means) / normalise_std
        
    # We set the color to the first channel.
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

"""
Predict the class or classes of an image using a pre-trained deep learning model.
@image_path string Path to the direct image
@model pytorch Neural Network (NN)
@top_k int The top K classes that will be calculated
@returns fn_top_probabilities(k) top_labels
"""

def fn_predict(image_tensor, model, device, cat_to_name, top_k):
    
    if type(top_k) == type(None):
        top_k = 5
        print("Top K was not specified. Assuming K=5.")
    
    # Set the model to evaluate.
    model.eval()

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(
        np.expand_dims(image_tensor, axis=0)).type(torch.FloatTensor)

    # Set the model to cpu
    model = model.cpu()

    # Find probabilities (results) by passing through the activation function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] # Due to GPU issues we doing this shortcut
    
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

"""
Print the probability of the class of image.
Converting the two lists to a dictionary to print on screen
"""

def fn_print_probability(probs, flowers):
    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, likelihood: {}%".format(j[1], math.ceil(j[0]*100)))
    


# ------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------- #
def main():
        
    args = fn_parser()
    
    # Load the names of the categories from the json file.
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    model = fn_load_checkpoint(args.checkpoint)
    
    image_tensor = fn_process_image(args.image)
    
    device = check_gpu(gpu_arg=args.gpu)
    
    top_probs, top_labels, top_flowers = fn_predict(image_tensor, model, 
                                                 device, cat_to_name,
                                                 args.top_k)
    fn_print_probability(top_flowers, top_probs)
    
if __name__ == "__main__": main()
