# AI Programming with Python Project
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# Running the Training Script
This script will train a new network on a dataset and save the model as a checkpoint.

# Running the Prediction Script

    NB! Please ensure that you created the trained_results directory first before running the script else your model will not be saved.

Uses a trained network to predict the class for an input image.

Set directory to save checkpoints: 

```python train.py data_dir --save_dir save_directory```

Choose architecture: 

```python train.py data_dir --arch "vgg13"```

Set hyperparameters: 
    
```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 5```
    
Use GPU for training: 
    
```python train.py data_dir --gpu```

Predict flower name from an image with predict.py along with the probability of that name. 

That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage:
```python predict.py /path/to/image checkpoint image_path = "flowers/test/10/image_07090.jpg"```
    
Options:
Return top KKK most likely classes: 
```python predict.py input checkpoint --top_k 3```
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json

Use GPU for inference: 
```python predict.py input checkpoint --gpu```

# Training Commands :

```python train.py --save_dir train_results --gpu --learning_rate 0.001 --hidden_units 512 --epochs 5```

# Prediction commands :

```python predict.py flowers/test/10/image_07090.jpg train_results/checkpoint.pth --top_k 3```

# Training Parameters
    --use_gpu Sets the GPU
    --save_dir Save the trained model
    --learning_rate Learning rate that the model is trained at

