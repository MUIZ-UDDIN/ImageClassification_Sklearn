# Image Classification using SVM

## Overview

This project focuses on image classification using Support Vector Machines (SVM), a popular machine learning algorithm. The goal is to accurately classify images into two categories: "empty" and "not_empty." The SVM model is trained on a dataset of resized images, and its performance is evaluated on a test set.

## Data Preparation

The dataset is organized into two categories: "empty" and "not_empty." Images from each category are loaded, resized to a common size (15x15 pixels), flattened, and added to the training data (`Data`). Corresponding labels (`Labels`) indicate the category (0 for "empty" and 1 for "not_empty").

## Train/Test Split

The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn. The split is performed with 80% of the data used for training and 20% for testing.

## Model Training

A Support Vector Machine (SVM) classifier is used for model training. Hyperparameter tuning is performed using Grid Search with a range of values for the parameters gamma and C. The best-performing model is selected based on the grid search results.

## Model Evaluation

The performance of the trained SVM model is evaluated on the test set. The accuracy score is calculated, indicating the percentage of correctly classified samples. The results are printed, providing insights into the model's effectiveness.

## Save Model

The best-performing SVM model is saved using the `pickle` library. The model is stored as a serialized file named "model.p" for future use.

Requirements
Python 3.x
Libraries: os, numpy, pickle, scikit-learn, skimage
License
This project is licensed under the MIT License.
