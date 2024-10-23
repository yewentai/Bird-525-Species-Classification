# Bird Species Classification with PyTorch and ResNet50

This project demonstrates how to perform bird species classification using PyTorch and the ResNet50 architecture. It includes downloading a dataset of bird species from Kaggle, preprocessing the data, defining data augmentation strategies, training a ResNet50 model, and evaluating its performance. This project aims to provide an end-to-end example of using transfer learning for image classification tasks.

## Project Structure

The project is divided into three main scripts:

1. `dataset_download_preprocessing.py` - Handles the downloading of the bird species dataset from Kaggle and performs initial data preprocessing steps.

2. `plot_result.py` - Contains functions for plotting training results such as accuracy and loss over epochs.

3. `pytorch-resnet50.py` - The main script where the model is defined, trained, and evaluated. This script also includes functions for data augmentation, dataloader creation, and visualization of model predictions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch and torchvision
- Kaggle API (for downloading the dataset)
- Matplotlib and Seaborn (for plotting)
- PIL, OpenCV, Albumentations (for image preprocessing and augmentation)
- Super Gradients (for additional training utilities)

### Data Source

The data used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species). However, this kaggle datasets was not maintained anymore. Please search '525 birds' on [Google Dataset Search](https://datasetsearch.research.google.com/).

## Key Features

- **Data Augmentation**: Implements various data augmentation techniques to improve model robustness.
- **Transfer Learning**: Utilizes a pre-trained ResNet50 model and fine-tunes it for the task of bird species classification.
- **Model Evaluation**: Provides functions to evaluate the model's performance on a test set and visualize its predictions.
- **Custom Dataloader**: Demonstrates how to create custom PyTorch dataloaders for handling image data.

## Results

After training, the model's performance is evaluated on a test dataset. The script generates confusion matrices and plots sample predictions to visualize the model's accuracy.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The bird species dataset used in this project is available on Kaggle, provided by [gpiosenka](https://www.kaggle.com/gpiosenka/100-bird-species).
- This project utilizes the [Super Gradients](https://supergradients.com/) library for additional training utilities.
