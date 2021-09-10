# movie recommendation program

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# load data

df = pd.read_csv('New.csv')
print(df.head(3))
# df['id']=df.index
# df.to_csv('New')


# get count of movies in the data set and the number of columns
df.shape

# create list of important columns for the recommendation engine
columns = ['Actors', 'Director', 'Genre', 'Title']

print(df[columns].head(3))

df[columns].isnull().values.any()


# create function to combine values of the important columns

def get_important_values(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(
            data['Actors'][i] + ' ' + data['Director'][i] + ' ' + data['Genre'][i] + ' ' + data['Title'][i])

    return important_features


# Create a column to hold the combined strings
df['important_features'] = get_important_values(df)

# show data
print(df['important_features'].head(3))

# convert text to matirx of token counts

cm = CountVectorizer().fit_transform(df['important_features'])

# Get the cosine similarity matrix from the count matirx

cs = cosine_similarity(cm)
print(cs)

cs.shape

title = 'The Amazing Spider-Man'

movie_id = df[df.Title == title]['id'].values[0]

scores = list(enumerate(cs[movie_id]))

# sort list

sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
sorted_scores = sorted_scores[1:]

print(sorted_scores)

j = 0
print('The 7 most recommended movies to', title, 'are:\n')
for item in sorted_scores:
    movie_title = df[df.id == item[0]]['Title'].values[0]
    print(j + 1, movie_title)
    j = j + 1
    if j > 6:
        break
