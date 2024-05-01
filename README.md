# Leveraging the Power of Text Analytics to Inform Fire Managers

## Introduction
This repository contains the code for my MSBA capstone project for Spring 2024. This project was conducted on data provided by the University of Montana FireCenter from the WFDSS (Wildland Fire Decision Support System) database. It seeks to explore the applications of text analytics within the dataset. The primary applications examined were the potential to use classification modeling to find the text features most predictive of attributes like fire risk, acreage, and agency involvement. The code in this repository takes the data from its raw form, to a format suitable for analyisis, performs analysis, and exports data for visualization in Power BI.

## The Work

### Data Cleaning
I began by creating functions to rectify errors in the data across multiple coumns, and put in some quality checks for future iterations of the dataset. The function file oarrCleaning.py contains all functions used in this portion of the code.

Next, I moved on to processing the text. This was by far the most difficult part of data cleaning. The text was full of reppetitve entries that had been copied and pasted from previous reports. The text fields were also consistently left blank. I created functions to resolve this issue. During this process I also tokenize, case-fold, and remove generic and domain specific stop words from the text.

### Basic Descriptive Statistics
Once the data was cleaned, I generated basic descriptive statistics about the dataset. This primarily consisted of stats related to the text such as most common tokens and n-grams, notes field completion, unique tokens by text field, etc.

### Classifying Risk - Binary Classification
To begin exploring the possibilities of using classification modeling to derive insights, I built functions to classify, from vectorized text, a binary risk variable (high/other,mod/other, etc.). I utilized a penalized logistic regression model to classify the risk. This was succesful and I explore the results within the notebook.

### Regional Differences
Next I examined the differences in token usage and risk assignment across regions. These results were then output to csv for use in the Power BI visualization.

### Tokens of Interest
This section of the notebook allows for the user to input their own tokens that they would like to recieve insights about in the text. By simply changing the tokens in the target_words list, a user can modify this however they see fit.

### Sentiment Analyses
Utilizing the TextBlob sentiment analysis tools, I performed a variety of analyses to look for meaningful insights in how sentiment differs across fire attributes. This was largely unsuccesful, as the data is written in a very consistent, flat manner. Both polarity and subjectivity were examined. Results are visible in the notebook.

### Risk Classification - Binary & 3 Class
In this section of the notebook the model selection process for my risk classification is available. I try naive bayes, random forest, penalized logistic regression, stacker models, and more to determine the best models for classifying risk both as a binary and three class variable. Once the best models are determined I examine feature coefficients and interpret results.

### Risk Classification - N-Grams
In this section, I tested how using n-grams as training data compared to individual tokens. The results were fairly unimpressive.

### SVOs
On recommendation from a faculty mentor, I attempted to model using SVO pairings. The results for classification were not too exciting, but examining the disproportionate SVOs by risk was quite interesting.

### Final Model
This is the final iteration of the previously established risk classification models, with interpreted insights regarding the text features that drive prediction of risk.

### Agency Involvement
This section of the notebook deals with attempting to classify whether or not a fire management agency is involved in the fire. the results were surprisingly promising. Insights and model accuracies are all available within the notebook.

### Acreage
In this final section of the notebook I used acreage binned to quartiles to examine the text features that are predictive of different sized fires. I also perform a cosine analysis to determine text similarity between dires of different sizes.



## What are these files?


## Conclusion


