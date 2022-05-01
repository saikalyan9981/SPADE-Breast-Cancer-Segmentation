# SPADE-Breast-Cancer-Segmentation

## How to run the code

Models can be trained and tested using the provided experiment notebooks `Final_vanilla_UNet.ipynb`, `Final_spade_models.ipynb` located in the *Experiments* directory.

Detailed code implementation for the SPADE encoding blocks can be found under the *Models* directory. 

**Please make sure to install all the python packages specified in the `requirements.txt` file**.

The directory paths will need to be modified to reflect your local paths.

1) Final_vanilla_UNet.ipynb runs the UNet alone model. It only takes as input ROI images and produces predicted mask.
2) Final_spade_models.ipynb is the notebook that runs the models that take in as input ROI and context images. There are 4 configurations available. The configuration of choice can be selected by simply changing the variable 'experiment_name' in the fourth cell. The available configurations are

  a) experiment_name = 'res_spadeSCRes'
    The convolutional layers in the UNet encoder are Resnet implementation
    The contextual blocks are the output of two sequential SPADE blocks summed with the input context image
  
  b) experiment_name = 'res_spadeRes'
    The convolutional layers in the UNet encoder are Resnet implementation
    The contextual blocks are the output of two sequential SPADE blocks summed with the output of one SPADE block
  
  c) experiment_name = 'conv_spadeRes'
    The convolutional layers in the UNet encoder are the original UNet implemention with two sequential convolutions
    The contextual blocks are the output of two sequential SPADE blocks summed with the output of one SPADE block
  
  d) experiment_name = 'conv_spadeSCRes'
    The convolutional layers in the UNet encoder are the original UNet implemention with two sequential convolutions
    The contextual blocks are the output of two sequential SPADE blocks summed with the input context image
