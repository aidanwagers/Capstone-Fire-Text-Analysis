#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import math
import os
import random
import re
import warnings
from collections import Counter, defaultdict
from datetime import timedelta
from string import punctuation

import eli5
import ipywidgets as widgets
import janitor
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import scipy.stats as stats
import seaborn as sns
import spacy
import sqlite3
import winsound
import xgboost as xgb
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from IPython.display import display, clear_output
from nltk.corpus import stopwords, words
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from spellchecker import SpellChecker
from textblob import TextBlob
from wordcloud import WordCloud
from eli5.sklearn import PermutationImportance

# Configure warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
# Download necessary nltk data
nltk.download('punkt')
nltk.download('words')

# NLP setup
nlp = spacy.load('en_core_web_lg')
tb = TextBlob('')
nlp.add_pipe('spacytextblob')

# Stopwords for English
sw = stopwords.words('english')

# In[2]:
# Initialize vectorzer
vectorizer =TfidfVectorizer()

def train_logistic_reg_for_one_risk(text, 
                                    labels, 
                                    risk = 'high', 
                                    penalty='l2', 
                                    C=1, 
                                    max_iter=500):
    """
    Train a logistic regression model for classifying text data with binary risk variable.

    Parameters:
    text (list of str): List of text data.
    labels (list of str): List of labels corresponding to the text data.
    risk (str): The risk category to classify against. Default is 'high'.
    penalty (str): The norm penalty to apply ('l1' or 'l2'). Default is 'l2'.
    C (float): Inverse of regularization strength; must be a positive float. Default is 1.
    max_iter (int): Maximum number of iterations taken for the solvers to converge. Default is 500.

    Returns:
    tuple: A tuple containing the trained logistic regression model and the training and testing data splits.
           - Trained logistic regression model.
           - Training feature vectors.
           - Testing feature vectors.
           - Training labels.
           - Testing labels.
    """
    
    # Convert labels to binary: 1 for 'high risk' and 0 for 'other'
    labels_binary = labels.apply(lambda x: 1 if x == risk else 0)
    
    x = vectorizer.fit_transform(text)
    y = labels_binary.tolist()
    
    lrx_train, lrx_test, lry_train, lry_test = train_test_split(x, 
                                                                y, 
                                                                test_size=0.2,
                                                                random_state=2)
    
    model = LogisticRegression(penalty=penalty, 
                               C = C, 
                               random_state = 2, 
                               max_iter = max_iter)
    model.fit(lrx_train, lry_train)
    
    return model, lrx_train, lrx_test, lry_train, lry_test


# In[3]:


def risk_other_machine(df, target_risk):
    """
    Preprocesses a DataFrame by labeling one specified risk category as 'target_risk' and all others as 'other'.

    Parameters:
    df (DataFrame): The DataFrame containing the text data and risk labels.
    target_risk (str): The risk category to label as 'target_risk'.

    Returns:
    tuple: A tuple containing the preprocessed text data and labels.
           - Preprocessed text data (Series): Combined text data after preprocessing.
           - Preprocessed labels (Series): Risk labels after preprocessing.
    """
    copy = df.copy()    
    copy['rrf_rr_desc'] = copy['rrf_rr_desc'].apply(lambda x: 'other' 
                                                    if x != target_risk else x)
    copy_comb = copy['combined_text']
    copy_labels = copy['rrf_rr_desc']
    
    return copy_comb, copy_labels


# In[4]:


def train_nb_classifier(combined_texts=None, 
                        risk_labels=None, 
                        test_size=0.2, 
                        alpha = 1.0, 
                        random_state=2):
    """
    Train a Naive Bayes classifier for classifying text data into risk categories.

    Parameters:
    combined_texts (list or Series): List or Series containing combined text data.
    risk_labels (list or Series): List or Series containing corresponding risk labels.
    test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
    alpha (float): Laplace smoothing parameter. Default is 1.0.
    random_state (int): Random seed for reproducibility. Default is 2.

    Returns:
    tuple: A tuple containing the trained Naive Bayes classifier and the training and testing data splits.
           - Trained Naive Bayes classifier.
           - Training feature vectors.
           - Testing feature vectors.
           - Training labels.
           - Testing labels.
    """
    
    vectorizer = TfidfVectorizer()

    x_nb = vectorizer.fit_transform(combined_texts)
    y_nb = risk_labels

    x_nb_train, x_nb_test, y_nb_train, y_nb_test = train_test_split(x_nb, 
                                                                    y_nb, 
                                                                    test_size = test_size, 
                                                                    random_state = random_state)

    nb_classifier = MultinomialNB(alpha= alpha)
    nb_classifier.fit(x_nb_train, y_nb_train)

    return nb_classifier, x_nb_train, x_nb_test, y_nb_train, y_nb_test


# In[5]:


def train_knn_classifier(combined_texts=None, 
                         risk_labels=None, 
                         neighbors = 6, 
                         test_size=0.2, 
                         random_state=2):
    """
    Train a k-Nearest Neighbors (kNN) classifier for classifying text data into risk categories.

    Parameters:
    combined_texts (list or Series): List or Series containing combined text data.
    risk_labels (list or Series): List or Series containing corresponding risk labels.
    neighbors (int): Number of neighbors to consider. Default is 6.
    test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
    random_state (int): Random seed for reproducibility. Default is 2.

    Returns:
    tuple: A tuple containing the trained kNN classifier and the training and testing data splits.
           - Trained kNN classifier.
           - Training feature vectors.
           - Testing feature vectors.
           - Training labels.
           - Testing labels.
    """
    x = vectorizer.fit_transform(combined_texts)
    y = risk_labels.tolist()
    xnn_train, xnn_test, ynn_train, ynn_test = train_test_split(x,
                                                                y,
                                                                test_size = 0.2, 
                                                                random_state = 2)
    knn_fire = KNeighborsClassifier(n_neighbors = neighbors)
    knn_fire.fit(xnn_train, ynn_train)
    
    return knn_fire, xnn_train, xnn_test, ynn_train, ynn_test

def neighbor_number_search(xnn_train, xnn_test, ynn_train, ynn_test, min, max):
    """
    Perform a search for the optimal number of neighbors (k) for k-Nearest Neighbors (kNN) classification
    by training multiple models with different numbers of neighbors and plotting their accuracies.

    Parameters:
    xnn_train: Training feature vectors.
    xnn_test: Testing feature vectors.
    ynn_train: Training labels.
    ynn_test: Testing labels.
    min_neighbors (int): Minimum number of neighbors to search.
    max_neighbors (int): Maximum number of neighbors to search.

    Returns:
    None, this function plots the accuracy on training and testing data for different numbers of neighbors.
    """
    n_rng = np.arange(min,max)
    train_acc = np.empty(len(n_rng))
    test_acc = np.empty(len(n_rng))
    
    for i, k in enumerate(n_rng):
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(xnn_train, ynn_train)
        
        train_acc[i] = knn.score(xnn_train, ynn_train)
        test_acc[i] = knn.score(xnn_test, ynn_test)
    
    plt.plot(n_rng, test_acc, label = 'Accuracy on Testing Data')
    plt.plot(n_rng, train_acc, label = 'Accuracy on Training Data')
    plt.legend()
    plt.xlabel('# of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    


# In[6]:


def train_logistic_reg(text, labels, penalty = 'l2', C = 1, max_iter = 500):
    """
    Train a logistic regression classifier using the provided text data and labels.

    Parameters:
    text (list): A list of strings representing the text data.
    labels (list): Labels corresponding to the text data.
    penalty (str, default='l2'): The norm used in the penalization.
    C (float, default=1): Inverse of regularization strength; smaller values specify stronger regularization.
    max_iter (int, default=500): Maximum number of iterations for optimization.

    Returns:
    tuple: A tuple containing the trained logistic regression model and related data splits.
        - model (LogisticRegression): The trained logistic regression model.
        - lrx_train: Training feature vectors.
        - lrx_test: Testing feature vectors.
        - lry_train: Training labels.
        - lry_test: Testing labels.
    """
    x = vectorizer.fit_transform(text)
    y = labels.tolist()
    
    lrx_train, lrx_test, lry_train, lry_test = train_test_split(x, 
                                                                y, 
                                                                test_size = 0.2, 
                                                                random_state = 2)
    
    model = LogisticRegression(penalty = penalty, 
                               C = C, 
                               random_state = 2, 
                               max_iter = max_iter)
    model.fit(lrx_train,lry_train)
    
    return(model, lrx_train, lrx_test, lry_train, lry_test)
    
    
    


# In[7]:


def train_random_forest(text,labels,n_estimators = 500):
    """
    Train a random forest classifier using the provided text data and labels.

    Parameters:
    text (list): A list of strings representing the text data.
    labels (list): Labels corresponding to the text data.
    n_estimators (int, default=500): The number of trees in the forest.

    Returns:
    tuple: A tuple containing the trained random forest classifier and related data splits.
        - rf_classifier (RandomForestClassifier): The trained random forest classifier.
        - rfx_train: Training feature vectors.
        - rfx_test: Testing feature vectors.
        - rfy_train: Training labels.
        - rfy_test: Testing labels.
    """
    x = vectorizer.fit_transform(text)
    y = labels.tolist()
    
    rfx_train, rfx_test, rfy_train, rfy_test = train_test_split(x, 
                                                                y, 
                                                                test_size = 0.2, 
                                                                random_state = 2)
    rf_classifier = RandomForestClassifier(n_estimators= n_estimators, 
                                           random_state = 2)
    rf_classifier.fit(rfx_train, rfy_train)
    
    return(rf_classifier,rfx_train, rfx_test, rfy_train, rfy_test)

def best_random_forest(text, labels, n_estimators=500):
    """
    Perform hyperparameter tuning for a Random Forest classifier using GridSearchCV.

    Parameters:
    - text (list): The input text data.
    - labels (list): The target labels.
    - n_estimators (int, optional): The number of trees in the forest. Defaults to 500.

    Returns:
    Tuple: A tuple containing the best Random Forest classifier model, training data (features), testing data (features), training labels,       and testing labels.
    """
    
    x = vectorizer.fit_transform(text)
    y = labels.tolist()
    
    rfx_train, rfx_test, rfy_train, rfy_test = train_test_split(x, 
                                                                y, 
                                                                test_size=0.2, 
                                                                random_state=2)
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, 
                                           random_state=2)


    param_grid = {
        'n_estimators': [100, 500, 1000, 2000],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20, 30],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(rf_classifier, 
                               param_grid, 
                               cv=5, 
                               scoring='accuracy', 
                               n_jobs=-1)
    grid_search.fit(rfx_train, rfy_train)

    # Get the best model from grid search
    best_rf_classifier = grid_search.best_estimator_

    return best_rf_classifier, rfx_train, rfx_test, rfy_train, rfy_test


# In[8]:


def train_binary_classifier(x_train, y_train, class_label):
    """
    Train a binary classifier using One-vs-Rest strategy.

    Parameters:
        x_train: Training features.
        y_train: Training labels.
        class_label: The class label for binary classification.

    Returns:
        OneVsRestClassifier: Trained binary classifier model.
    """
    model = OneVsRestClassifier(LogisticRegression())
    model.fit(x_train, (y_train == class_label).astype(int))
    return model


# In[9]:


def try_variety_of_models(x_train_tfidf, y_train, x_test_tfidf, y_test):
    """
    Try a variety of models and report the results.

    Parameters:
    - x_train_tfidf: TF-IDF transformed training data.
    - y_train: Training labels.
    - x_test_tfidf: TF-IDF transformed test data.
    - y_test: Test labels.

    Returns:
    - Dictionary containing the classification reports for each model.
    """
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC()
    }

    results = {}

    for model_name, model in models.items():
        # Perform cross-validation on the training data
        scores = cross_val_score(model, x_train_tfidf, y_train, cv=5)

        # Fit the model on the training data
        model.fit(x_train_tfidf, y_train)

        # Make predictions on the test data
        predictions = model.predict(x_test_tfidf)

        # Generate classification report
        report = classification_report(y_test, predictions)

        # Store the results
        results[model_name] = {
            'Cross-Validation Scores': scores,
            'Test Set Classification Report': report
        }

    return results


# In[10]:


def predict_fire_size_and_find_terms(df, size_categories):
    """
    Predicts fire size category ('very_small' vs 'large') and identifies the most influential terms.
    
    Parameters:
    - df: DataFrame containing the fire reports and their size categories.
    - size_categories: List of size categories to include in the analysis (e.g., ['very_small', 'large']).
    """
    
    # Filter data for the specified size categories
    filtered_data = df[df['size_cat'].isin(size_categories)].copy()
    
    # Create a binary target variable
    filtered_data['target'] = (filtered_data['size_cat'] == size_categories[1]).astype(int)
    
    # Extract texts and labels
    texts = filtered_data['combined_text'].tolist()
    labels = filtered_data['target'].values
    
    # Vectorize texts
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        labels, 
                                                        test_size=0.2, 
                                                        random_state=2)
    
    # Initialize and fit classifier
    classifier = LogisticRegression(random_state=2)
    classifier.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, 
                                y_pred, 
                                target_names = size_categories))
    
    # Get feature names and coefficients from the model
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    
    # Match coefficients with feature names and sort
    feature_importance = sorted(zip(coefficients, feature_names), 
                                reverse=True)
    
    # Print the most influential terms for the first category in size_categories
    print(f"Most influential terms for '{size_categories[1]}' fires:")
    for coef, feature in feature_importance[:15]:
        print(f"{feature}: {coef}")
    
    # Print the most influential terms for the second category in size_categories
    print(f"\nMost influential terms for '{size_categories[0]}' fires:")
    for coef, feature in reversed(feature_importance[-15:]):
        print(f"{feature}: {coef}")

def store_coefficients(vectorizer, 
                       plr_model, 
                       num_features=15, 
                       filename='coefficients_across_classes.csv'):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = plr_model.coef_

    # Identify the top and bottom 15 features for each class
    top_bottom_features = set()
    for class_index in range(len(plr_model.classes_)):
        top_indices = np.argsort(coefficients[class_index])[-num_features:][::-1]
        bottom_indices = np.argsort(coefficients[class_index])[:num_features]
        top_bottom_features.update(top_indices)
        top_bottom_features.update(bottom_indices)

    # Convert set to list to index features
    top_bottom_feature_indices = list(top_bottom_features)

    # Prepare to write to CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header row with class names for coefficients
        header = ['Feature'] + [f'Coefficient_{class_name}' 
                                for class_name in plr_model.classes_]
        writer.writerow(header)

        # Write the coefficients for each top and bottom feature across all classes
        for idx in top_bottom_feature_indices:
            feature = feature_names[idx]
            # Retrieve the coefficient of this feature across all classes
            coeffs_across_classes = [coefficients[class_index][idx] for class_index 
                                     in range(len(plr_model.classes_))]
            row = [feature] + [f'{coeff:.4f}' for coeff in coeffs_across_classes]
            writer.writerow(row)
# In[ ]:

def store_predict_fire_size_and_find_terms(df, size_categories):
    """
    Predicts fire size category ('very_small' vs 'large') and identifies the most influential terms.
    
    Parameters:
    - df: DataFrame containing the fire reports and their size categories.
    - size_categories: List of size categories to include in the analysis (e.g., ['very_small', 'large']).
    - output_folder: Folder path where the output CSV files will be saved.
    """
    
    # Filter data for the specified size categories
    filtered_data = df[df['size_cat'].isin(size_categories)].copy()
    
    # Create a binary target variable
    filtered_data['target'] = (filtered_data['size_cat'] 
                               == size_categories[1]).astype(int)
    
    # Extract texts and labels
    texts = filtered_data['combined_text'].tolist()
    labels = filtered_data['target'].values
    
    # Vectorize texts
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        labels, 
                                                        test_size=0.2, 
                                                        random_state=2)
    
    # Initialize and fit classifier
    classifier = LogisticRegression(random_state=2)
    classifier.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test,
                                               y_pred, 
                                               target_names = size_categories, 
                                               output_dict=True)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, 
                                y_pred, 
                                target_names = size_categories))
    
    # Save classification report to CSV
    pd.DataFrame(classification_rep).transpose().to_csv("classification_report.csv")
    
    # Get feature names and coefficients from the model
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    
    # Match coefficients with feature names and sort
    feature_importance = sorted(zip(coefficients, feature_names), reverse=True)
    
    # Split features into positive and negative coefficients
    positive_features = [(coef, feature) for coef, feature in feature_importance if coef > 0]
    negative_features = [(coef, feature) for coef, feature in feature_importance if coef < 0]
    
    # Save influential features to separate CSV files
    pd.DataFrame(positive_features, columns=['Coefficient', 
                                             'Feature']).to_csv("positive_influential_features.csv", 
                                                                index=False)
    pd.DataFrame(negative_features, columns=['Coefficient', 
                                             'Feature']).to_csv("negative_influential_features.csv", 
                                                                index=False)

    # Print the most influential terms for the first category in size_categories
    print(f"Most influential terms for '{size_categories[1]}' fires:")
    print(pd.DataFrame(positive_features, columns=['Coefficient', 
                                                   'Feature']).head(15))
    
    # Print the most influential terms for the second category in size_categories
    print(f"\nMost influential terms for '{size_categories[0]}' fires:")
    print(pd.DataFrame(negative_features, columns=['Coefficient', 
                                                   'Feature']).head(15))


