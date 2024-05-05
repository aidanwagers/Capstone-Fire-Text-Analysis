#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
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
from datetime import datetime


# In[5]:

def cl_names(df):
    """
    Clean column names of a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to clean column names for.

    Returns:
    DataFrame: The DataFrame with cleaned column names.
    """
    return df.clean_names()
        
def OARR_Clean_and_Check(df):
    """
    Perform data checks on the input DataFrame according to specific criteria.

    Parameters:
    df (DataFrame): The DataFrame to be cleaned and checked.

    Returns:
    None
    """
    
    # There is one erroneous entry in the geo_are_name column, so I fix it here.
    df['geo_area_name'] = df['geo_area_name'].replace('1', 'Northwest')
    
    # Below I check to make sure all geo_area_names are properly filled and formatted.
    df['geo_area_name'] = df['geo_area_name'].astype(str)
    inc = df[df['geo_area_name'].str.len() < 3]
    inc2 = df[df['geo_area_name'].str.len() > 19]
    na = df[df['geo_area_name'].str.contains(r'[^a-zA-Z\s]')]
    
    if not inc.empty:
        print("The row(s) at below index(es) need attention. The region is likely incorrect")
        for index, row in inc.iterrows():
            print(index)
    if not na.empty:
        print("The row(s) at below index(es) need attention. The region is likely incorrect")
        for index, row in na.iterrows():
            print(index)
    if not inc2.empty:
        print("The row(s) at below index(es) need attention. The region is likely incorrect")
        for index, row in na.iterrows():
            print(index)
            
    # Check to make sure each entry has a unique identifier.
    if df['wfdss_org_id'].nunique() != len(df['wfdss_org_id']):
        print("At least one WFDSS_ORG_ID is not unique. This is worth investigating.")
    
    # Here I check the fire_name column.
    fire_names = df['fire_name'].astype(str)
    
    if fire_names.isnull().any():
        null_indices = fire_names.index[fire_names.isnull()]
        print(f"The row(s) at index {null_indices} have null values in 'fire_name'.")
    else:
        if fire_names.str.contains('nan').any():
            nan_indices = fire_names.index[fire_names.str.contains('nan')]
            print(f"The row(s) at index {nan_indices} have 'nan' string values in 'fire_name'.")
    
    # Below I use a simple check to determine there are no errors in the lat/long columns.
    latnlong = ('latitude','longitude')        
    
    for column_name in latnlong:
        dtype = df[column_name].dtype
        if dtype != 'float64':
            print(f"Datatype of '{column_name}' is incorrect.")

        numer_check = pd.to_numeric(df[column_name], errors='coerce').notnull().all()
        if not numer_check:
            print(f"All values in '{column_name}' are not numeric")

    # A check to ensure a 1:1 relationship.
    onecheck = df.groupby('geographic_area')['geo_area_name'].nunique()
    dty = df['geographic_area'].dtype

    if dty != int:
        df['geographic_area'] = df['geographic_area'].astype(int)

    if (onecheck > 1).any():
        print("One or more geo area number associates with more than 1 region. That's probably not right.")
    
    # Some date checking
    df['start_date_time'] = df['start_date_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d:%H%M"))
    
    for col, valid_range in [('start_date_time', None), ('start_year', range(2011, 2023)),
                         ('start_month', range(1, 13)), ('start_month_day_year', None)]:
        for idx, value in enumerate(df[col], start=1):
            if pd.isnull(value):
                if col == 'start_month_day_year':
                    print(f"Null value found in '{col}' at index {idx}")
                else:
                    print(f"Null values found in '{col}', investigate")
            elif valid_range is not None and value not in valid_range:
                print(f"Invalid value '{value}' found in '{col}' at index {idx}")
    


# In[ ]:




