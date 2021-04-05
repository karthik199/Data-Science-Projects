# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:48:02 2021

@author: Karthik
"""

import os

os.chdir("H:\Data Science\Assignments\Movie Recommandation")


import pandas as pd
import numpy as np
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")
credits.head()

movies.head()

print("Credits:",credits.shape)
print("Movies Dataframe:",movies.shape)


credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_merge = movies.merge(credits_column_renamed, on='id')
print(movies_merge.head())

movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
print(movies_cleaned.head())
print(movies_cleaned.info())
print(movies_cleaned.head(1)['overview'])

df = movies_cleaned
col_lst = df.columns

#df['overview'].dropna(axis =1)

df.dropna(axis = 1, inplace =True)







from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'])
print(tfv_matrix)
print(tfv_matrix.shape)










from sklearn.metrics.pairwise import sigmoid_kernel

# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
print(sig[0])






