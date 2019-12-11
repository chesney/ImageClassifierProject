# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Running the Training Script

The train.py script will train a new network on a dataset and save the model as a checkpoint.

NB! Please ensure that you created the trained_results directory first before running the script else your model will not be saved.

## Training Parameters

    --use_gpu Sets the GPU
    --save_dir Save the trained model to a folder
    --learning_rate Learning rate that the model is trained at

## Training the Neural Network

Run the below command to train the Neural Network.

```python train.py --save_dir train_results --gpu --learning_rate 0.001 --hidden_units 512 --epochs 5```

Additional parameters can be passed to the above command. Please see below for context.

### Set directory to save checkpoints: 

```python train.py data_dir --save_dir save_directory```

### Choose architecture: 
Setting the architecture as ```vgg16```
```python train.py data_dir --arch "vgg16"```

### Set hyperparameters: 
    
```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 5```
    
### Use GPU for training: 
    
```python train.py data_dir --gpu```

# Running the Prediction Script
 
The prediction.py script uses a trained network to predict the class for an input image.

We predict a flower name from an image using predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and the script will return the flower name and class probability.

### Basic usage:

Return top KKK most likely classes for the given image, using
a mapping of categories to real names:

```python predict.py --image flowers/test/10/image_07090.jpg --checkpoint ./train_results/checkpoint.pth --top_k 3 --category_names cat_to_name.json```

This command can be run on the GPU or CPU depending on which
device is available. Use the ```--gpu``` flag to run the above
command on the GPU.
