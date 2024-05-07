#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import standard libraries.
import csv
import math
import os
import random
import re
import warnings
from collections import (Counter, 
                         defaultdict)
from datetime import timedelta
import pprint
from random import randint
from string import punctuation

# Third-party library imports
import eli5
import importlib
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
import xgboost as xgb
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from imblearn.over_sampling import RandomOverSampler
from IPython.display import (display,
                             clear_output)
from nltk.corpus import (stopwords,
                         words)
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from scipy.sparse import hstack
from scipy.stats import (kruskal,
                         pearsonr)
from sklearn.decomposition import PCA
from sklearn.ensemble import (BaggingClassifier,
                              RandomForestClassifier,
                              StackingClassifier)
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix,
                             f1_score,
                             precision_recall_fscore_support,
                             precision_score,
                             recall_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (cross_val_score,
                                     GridSearchCV,
                                     RandomizedSearchCV, 
                                     train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (LabelEncoder,
                                   StandardScaler)
from sklearn.svm import SVC
from spellchecker import SpellChecker
from textblob import TextBlob
from wordcloud import WordCloud


# Setup nltk
sw = stopwords.words('english')
nlp = spacy.load('en_core_web_lg')
tb = TextBlob('')
# In[2]:

tfidf_vectorizer = TfidfVectorizer()

def get_most_common_words(texts, n_top=10):
    """
    Get the most common words from a list of texts.

    Parameters:
    texts (list of str): A list of texts from which to extract common words.
    n_top (int, optional): Number of top common words to return. Default is 10.

    Returns:
    list of tuple: A list of tuples containing the most common words and their frequencies.
        Each tuple is formatted as (word, frequency), where word is a string and frequency
        is an integer representing the number of occurrences of that word across all texts.
    """
    
    vectorizer = CountVectorizer(stop_words='english')
    word_matrix = vectorizer.fit_transform(texts)
    sum_words = word_matrix.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n_top]


# In[3]:


def analyze_agency_involvement(data, agency_list):
    """
    Analyze agencies' involvement in text data, train a model to predict involvement, and assess model performance.

    Parameters:
    data (DataFrame): A DataFrame containing text data and agency involvement indicators.
    agency_list (list of str): A list of agency names to analyze.

    Returns:
    Results: A dictionary where keys are agency names and values are dictionaries containing analysis results.
        Each agency dictionary contains the following keys:
        - 'involvement_percentage': Percentage of data instances involving the agency.
        - 'text_analysis': A dictionary containing various text analysis results:
            - 'common_words_present': List of common words when the agency is present.
            - 'common_words_absent': List of common words when the agency is absent.
            - 'relative_freq_present': Relative frequency of common words when the agency is present.
            - 'relative_freq_absent': Relative frequency of common words when the agency is absent.
            - 'sorted_comparative_freq': Sorted comparative frequency of words.
        - 'model_performance': A dictionary containing performance metrics of the logistic regression model:
            - 'accuracy': Accuracy score of the model.
            - 'precision': Precision score of the model.
            - 'recall': Recall score of the model.
            - 'f1': F1 score of the model.
        - 'feature_importance': A dictionary containing important features for the logistic regression model:
            - 'positive': Top positive features.
            - 'negative': Top negative features.
    """
    results = {}
    
    for agency in agency_list:
        agency_data = {
            'involvement_percentage': (data[agency].mean()) * 100,
            'text_analysis': {},
            'model_performance': {},
            'feature_importance': {}
        }
        
        # Splitting text based on agency involvement
        agency_present_text = data[data[agency] == 1]['combined_text'].dropna()
        agency_absent_text = data[data[agency] == 0]['combined_text'].dropna()
        
        # Common words analysis
        common_words_present = get_most_common_words(agency_present_text)
        common_words_absent = get_most_common_words(agency_absent_text)
        
        # Relative frequency analysis
        total_words_present = sum(freq for _, freq in common_words_present)
        total_words_absent = sum(freq for _, freq in common_words_absent)
        relative_freq_present = {word: freq / total_words_present for word, 
                                 freq in common_words_present}
        relative_freq_absent = {word: freq / total_words_absent for word, 
                                freq in common_words_absent}
        
        # Comparative frequency
        comparative_freq = {word: relative_freq_present.get(word, 0)\
                            - relative_freq_absent.get(word, 0)\
                            for word in set(relative_freq_present) | set(relative_freq_absent)}
        sorted_comparative_freq = sorted(comparative_freq.items(), 
                                         key=lambda x: abs(x[1]), 
                                         reverse=True)
        
        agency_data['text_analysis'] = {
            'common_words_present': common_words_present,
            'common_words_absent': common_words_absent,
            'relative_freq_present' : relative_freq_present,
            'relative_freq_absent' : relative_freq_absent,
            'sorted_comparative_freq': sorted_comparative_freq
        }
        
        # Model training and evaluation
        X = tfidf_vectorizer.fit_transform(data['combined_text'].fillna(''))
        y = data[agency]
        
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=0.2, 
                                                            random_state=42)
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, y_train)
        
        y_pred = lr_model.predict(X_test)
        
        # Storing model performance
        agency_data['model_performance'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Feature importance
        feature_names = tfidf_vectorizer.get_feature_names_out()
        coefficients = lr_model.coef_[0]
        features_df = pd.DataFrame({'Feature': feature_names, 
                                    'Coefficient': coefficients})
        features_sorted = features_df.sort_values(by='Coefficient', 
                                                  ascending=False)
        
        agency_data['feature_importance'] = {
            'positive': features_sorted.head(10).to_dict('records'),
            'negative': features_sorted.tail(10).to_dict('records')
        }
        
        # Store results for current agency
        results[agency] = agency_data
    
    return results


# In[4]:


def get_agency_stats(df, agency = 'nps'):
    """
    Get statistics and analysis results for a specific agency from a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    agency (str): The name of the agency to extract statistics for. Default is 'nps'.

    Returns:
    dictionary: A dictionary containing various statistics and analysis results for the specified agency.
        Keys in the dictionary include:
        - 'involvement_percentage': Percentage of reports involving the agency.
        - 'common_words_present': List of common words when the agency is present.
        - 'common_words_absent': List of common words when the agency is absent.
        - 'relative_freq_present': Relative frequency of common words when the agency is present.
        - 'relative_freq_absent': Relative frequency of common words when the agency is absent.
        - 'sorted_comparative_freq': Sorted comparative frequency of words.
        - 'accuracy': Accuracy score of the logistic regression model for the agency.
        - 'precision': Precision score of the logistic regression model for the agency.
        - 'recall': Recall score of the logistic regression model for the agency.
        - 'f1': F1 score of the logistic regression model for the agency.
        - 'positive': Top positive features according to the logistic regression model.
        - 'negative': Top negative features according to the logistic regression model.
        
    agency_rf_pres: List of highest relative frequency when the agency is present.

    agency_rf_abs: List of highest relative frequency when the agency is absent.
    """
    
    f_rows = df[df['agency'] == agency]
    
    agency_rf_pres = f_rows['relative_freq_present'].iloc[0]
    agency_rf_abs = f_rows['relative_freq_absent'].iloc[0]
    agency_involvement_percentage = f_rows['involvement_percentage']
    agency_common_words_present = f_rows['common_words_present']
    agency_common_words_absent = f_rows['common_words_absent']
    agency_relative_freq_present = f_rows['relative_freq_present']
    agency_relative_freq_absent = f_rows['relative_freq_absent']
    agency_sorted_comparative_freq = f_rows['sorted_comparative_freq']
    agency_accuracy = f_rows['accuracy']
    agency_precision = f_rows['precision']
    agency_recall = f_rows['recall']
    agency_f1 = f_rows['f1']
    agency_positive = f_rows['positive']
    agency_negative = f_rows['negative']

    # Return all extracted values
    return {
        'involvement_percentage': agency_involvement_percentage,
        'common_words_present': agency_common_words_present,
        'common_words_absent': agency_common_words_absent,
        'relative_freq_present': agency_relative_freq_present,
        'relative_freq_absent': agency_relative_freq_absent,
        'sorted_comparative_freq': agency_sorted_comparative_freq,
        'accuracy': agency_accuracy,
        'precision': agency_precision,
        'recall': agency_recall,
        'f1': agency_f1,
        'positive': agency_positive,
        'negative': agency_negative
    }
    return agency_rf_pres, agency_rf_abs

def pretty_print_results(results):
    """
    Print the results of agency statistics function in a formatted and readable manner.

    Parameters:
    results (dict): A dictionary containing agency statistics. This is used with the previously outputted dicts.

    Returns:
    None, prints to screen.
    """
    
    # Initialize PrettyPrinter
    pp = pprint.PrettyPrinter(indent=4)
    
    print("Agency Stats Overview:\n")
    
    # Table header
    print(f"{'Metric':<25} | {'Value'}")
    print("-" * 50)
    
    # Loop through each result, print formatted string for numbers 
    # and use pprint for complex types
    for key, value in results.items():
        if isinstance(value.iloc[0], (float, int)):
            # Format numbers to two decimal places if it's a float/int
            print(f"{key:<25} | {value.iloc[0]:.2f}")
        else:
            # Use pretty print for dictionaries, lists, etc.
            print(f"{key:<25} :")
            pp.pprint(value.iloc[0])
            print("-" * 50)

def adress_agency_imbalance(agency, df):
    """
    Remedies class imbalance for a specific agency using oversampling and trains a logistic regression model.

    Parameters:
    agency (str): The name of the agency for which class imbalance is to be addressed.
    df (DataFrame): The DataFrame containing the data.

    Returns:
    None, prints to screen.
    """
    
    # Initialize a vectorizer
    vectorizer = TfidfVectorizer(min_df=5, 
                                 max_df=0.9, 
                                 ngram_range=(1, 2), 
                                 stop_words='english')
    
    # Prepare the feature matrix and the target vector
    X = vectorizer.fit_transform(df['combined_text'].fillna(''))
    y = df[agency]
    
    # Addressing class imbalance
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    # Splitting the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, 
                                                        y_resampled, 
                                                        test_size=0.2, 
                                                        random_state=2)
    
    # Model training with hyperparameter tuning (if needed)
    lr_model = LogisticRegression(max_iter=1000, 
                                  class_weight='balanced')
    lr_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = lr_model.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Cross-validation for model evaluation
    cv_scores = cross_val_score(lr_model, 
                                X_resampled, 
                                y_resampled, 
                                cv=5, 
                                scoring='accuracy')  # Change scoring as needed
    print("Cross-validated scores:", cv_scores)
    print("Mean accuracy:", cv_scores.mean())
    
    # Feature importance analysis
    feature_names = vectorizer.get_feature_names_out()
    coefficients = lr_model.coef_[0]
    feature_coefficients = dict(zip(feature_names, coefficients))
    
    # Sorting feature coefficients for positive and negative classes
    sorted_positive_coefficients = sorted(feature_coefficients.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)[:12]
    sorted_negative_coefficients = sorted(feature_coefficients.items(),
                                          key=lambda x: x[1])[:12]
    
    # Printing sorted coefficients
    print("\nSorted positive class feature coefficients:")
    for feature, coefficient in sorted_positive_coefficients:
        print(f"{feature}: {coefficient}")
        
    print("\nSorted negative class feature coefficients:")
    for feature, coefficient in sorted_negative_coefficients:
        print(f"{feature}: {coefficient}")
        
        
# In[ ]:
def update_agency_stats_and_features(agency, df, 
                                     stats_file='agency_stats.csv', 
                                     features_file='agency_features.csv'):
    """
    Update statistics and feature coefficients for a specific agency and save them to or create CSV files.

    Parameters:
    agency (str): The name of the agency to update statistics and features for.
    df (DataFrame): The DataFrame containing the data.
    stats_file (str, optional): The filename to save/update statistics. Default is 'agency_stats.csv'.
    features_file (str, optional): The filename to save/update feature coefficients. Default is 'agency_features.csv'.

    Returns:
    None
    """
    
    vectorizer = TfidfVectorizer(min_df=5, 
                                 max_df=0.9, 
                                 ngram_range=(1,1), 
                                 stop_words='english')
    X = vectorizer.fit_transform(df['combined_text'].fillna(''))
    y = df[agency]
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled,
                                                        y_resampled,
                                                        test_size=0.2, 
                                                        random_state=2)
    
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, 
                                                                     y_pred, 
                                                                     average='weighted')
    
    # Load or create the statistics DataFrame
    try:
        stats_df = pd.read_csv(stats_file)
    except FileNotFoundError:
        stats_df = pd.DataFrame(columns=['Agency', 
                                         'Accuracy', 
                                         'Precision', 
                                         'Recall', 
                                         'F1 Score', 
                                         'Run Date'])
    
    new_stats = pd.DataFrame([{
        'Agency': agency,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Run Date': pd.Timestamp.now()
    }])
    stats_df = pd.concat([stats_df, new_stats], ignore_index=True)
    stats_df.to_csv(stats_file, index=False)
    
    # Feature coefficients extraction and update
    feature_names = vectorizer.get_feature_names_out()
    coefficients = lr_model.coef_[0]
    feature_data = pd.DataFrame({
        'Agency': [agency]*len(feature_names),
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Load or create the features DataFrame
    try:
        features_df = pd.read_csv(features_file)
    except FileNotFoundError:
        features_df = pd.DataFrame(columns=['Agency', 
                                            'Feature', 
                                            'Coefficient'])
    
    features_df = pd.concat([features_df, feature_data], 
                            ignore_index=True)
    features_df.to_csv(features_file, 
                       index=False)
    
    print(f"Updated statistics for {agency} saved to {stats_file}.")
    print(f"Feature coefficients for {agency} saved to {features_file}.")





# In[ ]:




