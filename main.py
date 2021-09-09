# movie recommendation program

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

#load data

df = pd.read_csv('IMDB-Movie-Data.csv')
print(df.head(3))



#get count of movies in the data set and the number of columns
df.shape
(1000, 13)

# create list of important columns for the recommendation engine
columns = ['Actors', 'Director', 'Genre', 'Title']

print(df[columns].head(3))


