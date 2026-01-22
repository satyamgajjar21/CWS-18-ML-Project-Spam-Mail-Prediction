# Spam Mail Prediction

A machine learning project that classifies emails as spam or not spam using natural language processing techniques.  
This project demonstrates a complete NLP text classification pipeline using Python and scikit learn.

## Table of Contents
1. Overview
2. Problem Statement
3. Dataset Description
4. Text Preprocessing
5. Feature Extraction
6. Machine Learning Model
7. Evaluation Metrics
8. How to Run the Project
9. Results
10. Technologies Used
11. Future Improvements
12. Author

## Overview
Spam email detection is a classic and practical application of natural language processing.  
This project builds a machine learning model that automatically identifies spam emails based on the content of the message.

The notebook covers text cleaning, feature extraction, model training, evaluation, and prediction.

## Problem Statement
Given the content of an email message, predict whether the email is spam or not spam.

The target variable contains two classes:
0 represents not spam  
1 represents spam

## Dataset Description
The dataset consists of email messages and their corresponding labels.

Each row represents one email  
One column contains the email text  
One column contains the label indicating spam or not spam

## Text Preprocessing
The following text preprocessing steps are applied:
1. Lowercasing text
2. Removing punctuation
3. Removing numbers
4. Removing emojis
5. Removing stop words
6. Tokenization

These steps help reduce noise and improve model performance.

## Feature Extraction
Text data is converted into numerical features using feature extraction techniques.

The primary method used is:
1. Term Frequency Inverse Document Frequency

TF IDF helps assign higher importance to meaningful words while reducing the impact of common words.

## Machine Learning Model
This project uses a supervised classification algorithm.

Multinomial Naive Bayes is used as the primary model because:
1. It works well with text based features
2. It is fast and efficient
3. It performs well for spam detection tasks

## Evaluation Metrics
Model performance is evaluated using:
1. Accuracy score
2. Confusion matrix
3. Precision and recall

These metrics help measure how well the model identifies spam emails.

## How to Run the Project
1. Clone the repository to your local system
2. Ensure Python version 3.8 or higher is installed
3. Install required libraries
4. Open the notebook using Jupyter Notebook or Jupyter Lab
5. Run all cells sequentially

## Results
The trained model successfully classifies emails into spam and not spam categories with good accuracy.

Results may vary depending on preprocessing choices and feature extraction parameters.

## Technologies Used
1. Python
2. NumPy
3. Pandas
4. Scikit learn
5. NLTK
6. Matplotlib
7. Seaborn
8. Jupyter Notebook

## Future Improvements
1. Try additional classifiers such as Logistic Regression or Support Vector Machines
2. Tune TF IDF parameters for better performance
3. Handle class imbalance if present
4. Use n grams for richer feature representation
5. Deploy the model as an API or web application

## Author
Satyam Gajjar
