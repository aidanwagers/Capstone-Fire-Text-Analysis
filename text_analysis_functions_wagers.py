#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


import nltk
from datetime import timedelta
import os
nltk.download('punkt')
from nltk.corpus import stopwords
import numpy as np
import sqlite3
from collections import Counter, defaultdict
from string import punctuation
import re
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
sw = stopwords.words('english')
import janitor
import spacy
from textblob import TextBlob
tb = TextBlob('')
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_lg')
import seaborn as sns
import scipy.stats as stats
from scipy.stats import kruskal
from gensim.models import CoherenceModel
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import ipywidgets as widgets
from ipywidgets import Layout
from ipywidgets import interact_manual, interactive_output
from IPython.display import display, clear_output
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import spacy
import math
from textblob import TextBlob
tb = TextBlob('')
from spacytextblob.spacytextblob import SpacyTextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from random import randint
from sklearn.model_selection import RandomizedSearchCV
import winsound
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from spellchecker import SpellChecker
from textblob import TextBlob
import eli5
from eli5.sklearn import PermutationImportance
import warnings
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import words
from nltk.util import ngrams


# In[2]:


def count_ngrams(text, n):
    """
    Count and sort n-grams in the given text to find the most common n-grams.

    Parameters:
    text (str): The text to be processed.
    n (int): The number specifying the size of n-grams to generate.

    Returns:
    sorted_n_grams: A dictionary containing the n-grams as keys and their frequencies as values, sorted in descending order of frequency.
    """
    tokens = word_tokenize(text)  
    n_grams = ngrams(tokens, n)
    n_gram_freq = Counter(n_grams)
    sorted_n_grams = dict(sorted(n_gram_freq.items(), key=lambda item: item[1], reverse=True))
     
    return sorted_n_grams

def completion_percentage(df, columns_to_process):
    """
    Calculate the completion percentage of text fields in the specified columns of the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the text data.
    columns_to_process (list of str): A list of column names to process.

    Returns:
    completion_dict: A dictionary where keys are column names and values are the completion percentages (in percentage) 
          of non-empty entries in the respective columns.
    """
    completion_dict = {}

    for column in columns_to_process:
        total_count = len(df)
        non_empty_count = df[column].apply(lambda x: x.strip() if isinstance(x, str) else x).replace('', pd.NA).notna().sum()
        
        if total_count > 0:
            perc_completion = non_empty_count / total_count * 100
        else:
            perc_completion = 0
        
        completion_dict[column] = perc_completion
        
    print("Below is the percantage of entires with at least some unique text for a fire-risk grouping:")
    return completion_dict

def basic_text(df, columns_to_process):
    """
    Perform basic text analyses for specified columns in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the text data.
    columns_to_process (list of str): A list of column names to process.

    Returns:
    results_df: A DataFrame containing basic text analysis results for each specified column.
               The DataFrame has columns for each input column and rows for the following analysis metrics:
               - total_tokens: Total number of tokens in the column.
               - unique_tokens: Number of unique tokens in the column.
               - average_tokens: Average number of tokens per entry in the column.
               - avg_token_length: Average length of tokens in the column.
               - lexical_diversity: Lexical diversity ratio (unique_tokens / total_tokens) in the column.
               - top_10: List of top 10 most common tokens and their frequencies in the column.
    """
    results = {}
    
    for column in columns_to_process:
        concatenated_text = df[column].astype(str).str.cat(sep=' ')

        text_clean = [word for word in concatenated_text.split()]
        c = Counter(text_clean)

        total_tokens = len(text_clean)
        unique_tokens = len(set(text_clean))

        lex_diversity = unique_tokens / total_tokens if total_tokens != 0 else "NA"
        avg_token_len = np.mean([len(word) for word in text_clean])
        top_10 = c.most_common(10)

        avg_tokens_per_entry = df[column].apply(lambda x: len(str(x).split())).mean()
        
        
        results[column] = {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'average_tokens' : avg_tokens_per_entry,
            'avg_token_length': avg_token_len,
            'lexical_diversity': lex_diversity,
            'top_10': top_10,
        }
    results_df = pd.DataFrame(results)
    return results_df


# In[3]:


def contains_target_word(text, target_word):
    """
    Check if the given text contains a target word.

    Parameters:
    text (str): The text to be analyzed.
    target_word (str): The word to search for in the text.

    Returns:
    bool: True if the target word is found in the text, False otherwise.
    """
    return target_word in text

def percentage_with_target_words(data, target_words):
    """
    Calculate the percentage of entries in the DataFrame containing the specified target words.
    Uses contains_target_word.

    Parameters:
    data (DataFrame): The DataFrame containing text data.
    target_words (list of str): A list of target words to search for.

    Returns:
    None, prints the percentage of entries containing each target word in the combined notes fields for fires.
    """
    percentage_with_target_words = {}

    data_copy = data.copy()

    for word in target_words:
        data_copy[f'contains_{word}'] = data_copy['combined_text'].apply(contains_target_word, target_word=word)
        percentage_with_target_words[word] = (data_copy[f'contains_{word}'].sum() / len(data_copy)) * 100
        data_copy.drop(columns=[f'contains_{word}'], inplace=True)

    sorted_results = sorted(percentage_with_target_words.items(), key=lambda x: x[1], reverse=True)

    for word, percentage in sorted_results:
        print(f"The word '{word}' appears in {percentage:.2f}% of the combined notes fields for fires.")
        
def percentage_by_risk(data, target_words):
    """
    Calculate the percentage of entries in the DataFrame containing the specified target words for each risk level.

    Parameters:
    data (DataFrame): The DataFrame containing text data.
    target_words (list of str): A list of target words to search for.

    Returns:
    None, prints the percentage of entries containing each target word for each risk level.
    """
    results_by_risk = {}

    data_copy = data.copy()

    custom_order = ['high', 'mod', 'low']

    grouped_data = data_copy.groupby('rrf_rr_desc', sort=False)  # Use sort=False to maintain custom order

    for risk, group in grouped_data:
        percentage_with_target_words = {}
        for word in target_words:
            group[f'contains_{word}'] = group['combined_text'].apply(contains_target_word, target_word=word)
            percentage_with_target_words[word] = (group[f'contains_{word}'].sum() / len(group)) * 100
            group.drop(columns=[f'contains_{word}'], inplace=True)
        results_by_risk[risk] = percentage_with_target_words

    for risk in custom_order:
        if risk in results_by_risk:
            percentages = results_by_risk[risk]
            print(f"Risk: {risk.capitalize()}")
            for word, percentage in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
                print(f"The word '{word}' appears in {percentage:.2f}% of the texts for this risk.")
            print()

            
            
            
def extract_context(text, target_word, context_window_size = 5):
    """
    Extract the context surrounding the target word in the text.

    Parameters:
    text (str): The text to analyze.
    target_word (str): The word to examine.
    context_window_size (int): The number of words to include before and after the target word in the context. Default is 5.

    Returns:
    contexts: A list of lists, where each inner list represents the context surrounding a target word.
                          Each inner list contains the words in the context window.
    """
    words = text.split()
    target_indices = [i for i, word in enumerate(words) if word.lower() == target_word]
    
    contexts = []
    for index in target_indices:
        start_index = max(0, index - context_window_size)
        end_index = min(len(words), index + context_window_size + 1)
        context = words[start_index:end_index]
        contexts.append(context)
    
    return contexts

def extract_ordered_ngrams(context_list, n =3):
    """
    Extract ordered n-grams from a list of contexts. Builds off previous function when used in notebook.

    Parameters:
    context_list (list of list of str): A list of lists, where each inner list represents a context containing words.
    n (int): The size of the n-grams to extract. Default is 3.

    Returns:
    ordered_trigrams: A list of ordered n-grams extracted from the context list.
    """
    ordered_trigrams = []
    for context in context_list:
        ordered_trigrams.extend(" ".join(context[i:i+n]) for i in range(len(context) - 2))
    return ordered_trigrams

def vectorize_text_with_ngrams(text_data, labels, ngram_range=(1, 2), test_size=0.2):
    """
    Vectorize text data using TF-IDF with specified n-gram range.

    Parameters:
    - text_data: List of text data.
    - labels: List of corresponding labels.
    - ngram_range: Tuple specifying the range of n-grams (default is (1, 2)).
    - test_size: Fraction of the dataset to be used as the test set (default is 0.2).
    - random_state: Seed for random number generation (default is None).

    Returns:
    - Tuple (train_tfidf, test_tfidf, y_train, y_test)
      - train_tfidf: TF-IDF transformed training data.
      - test_tfidf: TF-IDF transformed test data.
      - y_train: Training labels.
      - y_test: Test labels.
    """
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(text_data, labels, test_size=test_size, random_state= 2)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    return x_train_tfidf, x_test_tfidf, y_train, y_test


# In[4]:


def average_tfidf_vectors(tfidf_matrix, indices):
    """
    Computes the average TF-IDF vector for the given indices in the TF-IDF matrix.

    Parameters:
    tfidf_matrix (numpy.ndarray): The TF-IDF matrix.
    indices (list): List of indices to compute the average for.

    Returns:
    numpy.ndarray: The average TF-IDF vector.
    """
    selected_vectors = tfidf_matrix[indices]
    average_vector = np.mean(selected_vectors, axis=0)
    return average_vector


def button_click_handler(button):
    """
    Handle a button click event by updating an output area with a plot generated based on the button's description.

    Parameters:
    button (Button): The button object that triggered the click event.
    
    Returns:
    None
    """
    with output_area:
        clear_output(wait=True)
        plot_function(button.description)

        
def plot_function(word):
    """
    Plot a bar chart showing the common tokens appearing in context for the given word.

    Parameters:
    word (str): The word for which context data is plotted.

    Returns:
    None, plots chart.
    """
    plt.figure(figsize=(8, 6))
    
    # Ensure there is data for the given token in the dictionary
    if 'default_risk' in context_words_dict_alpha.get(word, {}):  # Updated from cwd
        common_words, counts = zip(*context_words_dict_alpha[word]['default_risk'])  # Updated from cwd
        plt.bar(common_words, counts)
        plt.title(f"Common Tokens appearing in Context for '{word}'")  # Updated title
        plt.xlabel("Words")
        plt.ylabel("Count")
        plt.show()
    else:
        print(f"No data available for combined text for {word}")
    plt.figure(figsize=(8, 6))
    
    # Ensure there is data for the given token in the dictionary
    if 'default_risk' in context_words_dict_alpha.get(word, {}):  # Updated from cwd
        common_words, counts = zip(*context_words_dict_alpha[word]['default_risk'])  # Updated from cwd
        plt.bar(common_words, counts)
        plt.title(f"Common Tokens appearing in Context for '{word}'")  # Updated title
        plt.xlabel("Words")
        plt.ylabel("Count")
        plt.show()
    else:
        print(f"No data available for combined text for {word}")

def plot_risks(word):
    """
    Plot bar charts showing the common tokens appearing in context for the given word across different risk descriptions.

    Parameters:
    word (str): The word for which context data is plotted.

    Returns:
    None, plots chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Ensure there is data for the given token in the dictionary
    if word in context_words_dict:
        # Sort risk descriptions based on counts
        sorted_risks = sorted(context_words_dict[word].items(), key=lambda x: sum(count for word, count in x[1]), reverse=True)

        for ax, (risk_description, common_words) in zip(axes, sorted_risks):
            words, counts = zip(*common_words)
            ax.bar(words, counts)
            ax.set_title(risk_description)
            ax.set_xlabel("Words")
            ax.set_ylabel("Count")

        plt.show()
    else:
        print(f"No data available for combined text for {word}")


def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text using TextBlob.

    Parameters:
    text (str): The text to analyze.

    Returns:
    tuple: A tuple containing the polarity and subjectivity of the sentiment analysis.
           Polarity ranges from -1 (negative) to 1 (positive), where 0 is neutral.
           Subjectivity ranges from 0 to 1, where 0 is very objective and 1 is very subjective.
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity


# In[ ]:





# In[ ]:




