1. **Clone the Repository**

   Clone the project repository from GitHub to your local machine.

   `git clone https://github.com/jaredtanzd/CZ4045.git`
   
   `cd CZ4045`

2. **Create and Activate the Virtual Environment**

   On Windows:

   `python -m venv nndl`
   
   `.\nndl\Scripts\activate`

   On macOS and Linux:

   `python -m venv nndl`
   
   `source nndl/bin/activate`

3.  **Install Dependencies**

    Use the requirements.txt file to install the necessary dependencies.

    `pip install -r requirements.txt`

## train.py
- This file holds the code used to train the baseline models for vgg16, resnet152 and mobilenetv2.
- `python train.py`

- This command will load the dataset from dataset.py into dataloaders and train the models, as well as save the model weights

## Few_Shot_Learning.ipynb
- This python notebook holds the code used to implement few shot learning
- Make sure the python interpreter points to the correct venv and simply run all the cells

## Advanced Transformation Techniques.ipynb
- This python notebook holds the code used to implement Cutmix and Mixup augmentation
- Make sure the python interpreter points to the correct venv and simply run all the cells

## Miscellaneous
- common_utils.py holds the code for EarlyStopper class
- dataset.py holds to code for generating Oxford Flowers 102 dataset
- dcn.py holds code for deformable convolution to be used in models
- model.py holds code for loading pretrained models and modifying fc/classifier layers, and implement deformable convolution
