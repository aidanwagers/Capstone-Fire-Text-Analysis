# Leveraging the Power of Text Analytics to Inform Fire Managers

## Introduction
This repository contains the code for my MSBA capstone project for Spring 2024. This project was conducted on data provided by the University of Montana FireCenter from the WFDSS (Wildland Fire Decision Support System) database. It seeks to explore the applications of text analytics within the dataset. The primary applications examined were the potential to use classification modeling to find the text features most predictive of attributes like fire risk, acreage, and agency involvement. The code in this repository takes the data from its raw form, to a format suitable for analysis, performs analysis, and exports data for visualization in Power BI.

## The Work

### Data Cleaning
I began by creating functions to rectify errors in the data across multiple columns, and put in some quality checks for future iterations of the dataset. The function file oarrCleaning.py contains all functions used in this portion of the code.

Next, I moved on to processing the text. This was by far the most difficult part of data cleaning. The text was full of repetitive entries that had been copied and pasted from previous reports. The text fields were also consistently left blank. I created functions to resolve this issue. During this process I also tokenize, case-fold, and remove generic and domain specific stop words from the text.

### Basic Descriptive Statistics
Once the data was cleaned, I generated basic descriptive statistics about the dataset. This primarily consisted of stats related to the text such as most common tokens and n-grams, notes field completion, unique tokens by text field, etc.

### Classifying Risk - Binary Classification
To begin exploring the possibilities of using classification modeling to derive insights, I built functions to classify, from vectorized text, a binary risk variable (high/other,mod/other, etc.). I utilized a penalized logistic regression model to classify the risk. This was successful and I explored the results within the notebook.

### Regional Differences
Next I examined the differences in token usage and risk assignment across regions. These results were then output to csv for use in the Power BI visualization.

### Tokens of Interest
This section of the notebook allows for the user to input their own tokens that they would like to receive insights about in the text. By simply changing the tokens in the target_words list, a user can modify this however they see fit.

### Sentiment Analyses
Utilizing the TextBlob sentiment analysis tools, I performed a variety of analyses to look for meaningful insights in how sentiment differs across fire attributes. This was largely unsuccessful, as the data is written in a very consistent, flat manner. Both polarity and subjectivity were examined. Results are visible in the notebook.

### Risk Classification - Binary & 3 Class
In this section of the notebook the model selection process for my risk classification is available. I try naive bayes, random forest, penalized logistic regression, stacker models, and more to determine the best models for classifying risk both as a binary and three class variable. Once the best models are determined I examine feature coefficients and interpret results.

### Risk Classification - N-Grams
In this section, I tested how using n-grams as training data compared to individual tokens. The results were fairly unimpressive.

### SVOs
On recommendation from a faculty mentor, I attempted to model using SVO pairings. The results for classification were not too exciting, but examining the disproportionate SVOs by risk was quite interesting.

### Final Model
This is the final iteration of the previously established risk classification models, with interpreted insights regarding the text features that drive prediction of risk.

### Agency Involvement
This section of the notebook deals with attempting to classify whether or not a fire management agency is involved in the fire. The results were surprisingly promising. Insights and model accuracies are all available within the notebook.

### Acreage
In this final section of the notebook I used acreage binned to quartiles to examine the text features that are predictive of different sized fires. I also perform a cosine analysis to determine text similarity between fires of different sizes.



## What are these files?
### Product Notebook.ipynb:
This is the jupyter notebook containing the final code for my capstone project. It is the synthesis of multiple other notebooks created during this project.

### agency_involvement_functions.py:
This .py file contains all the functions I made to work on gaining insight into agency involvement in the data. It contains the following functions:
* adress_agency_imbalance
* analyze_agency_involvement
* get_agency_stats
* get_most_common_words
* pretty_print_results
* update_agency_stats_and_features

### oarr_cleaning.py:
Contains two functions that are used for cleaning column names and checking data for errors. It contains the following functions:
* cl_names
* oarr_clean_and_check 

### text_analysis_functions_wagers.py:
Contains all text analysis functions unrelated to classification of text. It contains the following functions:
* analyze_sentiment
* average_tfidf_vectors
* basic_text
* button_click_handler
* completion_percentage
* contains_target_word
* count_ngrams
* extract_context
* extract_ordered_ngrams
* percentage_by_risk
* percentage_with_target_words
* plot_function
* vectorize_text_with_ngrams

### text_classification_functions.py:
Contains all functions related to classification of text. These include:
* best_random_forest
* neighbor_number_search
* predict_fire_size_and_find_terms
* risk_other_machine
* store_coefficients
* store_predict_fire_size_and_find_terms
* train_knn_classifier
* train_logistic_reg
* train_logistic_reg_for_one_risk
* train_nb_classifier
* train_random_forest
* try_variety_of_models

### text_functions.py:
This file contains functions used in processing the text to prepare it for anlysis. They are as follows:
* categorize_columns
* firestops
* get_last_value
* process_text_column
* set_columns_to_string

### three_ps:
This is a weekly update file to track my progress throughout the project lifecycle.

### README.md:
The file you are reading now!




