#!/usr/bin/env python
# coding: utf-8

# In[1]:
# CONTENT-BASED FILTERING
# In[2]:

# importing libraries
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# In[4]:

# storing the movies info
movies_df = pd.read_csv(r"C:\Users\admin\Downloads\movie recommender\ml-latest\movies.csv")
# storing ratings info
ratings_df = pd.read_csv(r"C:\Users\admin\Downloads\movie recommender\ml-latest\ratings.csv")
movies_df.head()


# In[7]:


# preprocessing data
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()


# In[8]:

movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()

# In[9]:


#Copying the movie dataframe into a new dataframe
moviesWithGenres_df = movies_df.copy()


# In[10]:


for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1

# filling NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()


# In[12]:


ratings_df.head()


# In[13]:


ratings_df = ratings_df.drop('timestamp',1)
ratings_df.head()


# In[14]:


# To add more movies, increase the amount of elements in the userInput.
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies


# In[15]:


#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

#Then merging it so we can get the movieId
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

#Final input dataframe
inputMovies


# In[16]:


#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies


#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
userGenreTable


inputMovies['rating']


# dot product to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
userProfile



#genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()


genreTable.shape


#Multiplying the genres by the weights and then taking weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()


#Sorting recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
recommendationTable_df.head()


#final recommendation table
movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
