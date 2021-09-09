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

df[columns].isnull().values.any()

#create function to combine values of the important columns

def get_important_values(data):
    important_features = []
    for i in range(0,data.shape[0]):
        important_features.append(data['Actors'][i]+' '+data['Director'][i] + ' '+data['Genre'][i]+' '+ data['Title'][i])

    return important_features


# Create a column to hold the combined strings
df['important_features'] = get_important_values(df)

#show data
print(df[columns].head(3))


#convert text to matirx of token counts

cm = CountVectorizer().fit_transform(df['important_features'])

#Get the cosine similarity matrix from the count matirx

cs = cosine_similarity(cm)



