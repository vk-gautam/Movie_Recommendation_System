import numpy as np
import pandas as pd
import warnings  # it is a package

warnings.filterwarnings('ignore')       # we can ignore any warnings usings this

#Loading Dataset 'Movielens'. It has 1 lakh rows, 4 columns and no column name. We need to give column name else first row will be included in column names
columns_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=columns_names)   # This is not a csv file but in this, data is separated by tabs so we need to pass separater as tab

#x = df['user_id'].nunique()     # Getting no. of unique user_id
#y = df['item_id'].nunique()     # Getting no. of unique movies
#print(x)
#print(y)

movies = pd.read_csv('ml-100k/u.item', sep='\|', header = None) # To get the movie title corresponding  to the item_id
#print(df) #It has a lot of columns but we need only first two i.e., item_id and movie_name with column names as 0 and 1
movie_titles = movies[[0,1]]        # getting item_id and movie_name
movie_titles.columns = ['item_id', 'title']
#print(movie_titles)

# Now we have two data frames and they have a common column calles item_id. So we can merge both the dataframes in one based on item_id
merged_df = pd.merge(df, movie_titles, on='item_id')
#print(merged_df)
#Exploitary Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')      # optional

# Now we need to check the average rating of each movie. For this we can group the movies by their names
# and find the mean. The mean function on dataframe will find the mean of each column but we need only rating one
# So we can extract it out
mean_df = merged_df.groupby('title').mean()['rating'].sort_values(ascending = False)    # Finding average rating of each movie
#print(mean_df)  #Movies with rating 5 has negligible probability.

# Now we check which movie is watched by how many persons
count_df = merged_df.groupby('title').count()['rating'].sort_values(ascending = False)
#print(count_df)

# Create a dataframe of mean ratings and also include count in it
ratings_df = pd.DataFrame(merged_df.groupby('title').mean()['rating'])
#print(ratings_df)

ratings_df['No. of ratings'] = pd.DataFrame(merged_df.groupby('title').count()['rating'])       # Adding one more column which is the number of rating (count)
#print(ratings_df)

sorted_df = ratings_df.sort_values(by='rating', ascending=False)
#print(sorted_df)

# Plotting histograms of No. of ratings and Ratings
#plt.figure(figsize=(10,6))
#plt.hist(ratings_df['No. of ratings'], bins = 70)
#plt.show()

#plt.hist(ratings_df['rating'], bins = 70)
#plt.show()

#Plotting a joint plot No. of ratings and Ratings
#sns.jointplot(x='rating', y='No. of ratings', data = ratings_df, alpha = 0.5)
#plt.show()

########## CREATING MOVIE RECOMMENDATION ###########

# We'll make a matrix whose row(index) will represent user_id and column will represent title and 
# each cell will represent the rating given by that user to that movie
# We can make matrix from scratch but pandas has a method pivot_table for it.
# NaN in the cell represents that the person has not rated that movie may be because he/she has not watched that movie

moviemat = merged_df.pivot_table(index='user_id', columns='title', values='rating')
#print(moviemat)

#Checking highly watched movie
#x=ratings_df.sort_values('No. of ratings', ascending=False)
#print(x)    # Star Wars (1977) comes out to be highly watched

# Getting userwise rating of Star Wars (1977)

starwars_user_ratings = moviemat['Star Wars (1977)']
#print(starwars_user_ratings)  # starwars_user_ratings is a series not a dataframe. We can co-relate this series with the matrix moviemat. So it will give corelation of each movie

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)  # It will give correlation of Star Wars (1977) with each movie.
#print(similar_to_starwars)    #It is a series. We have convert it into DataFrame
corr_starwars = pd.DataFrame(similar_to_starwars, columns = ['Correlation'])        # Named the column having correlation values as Correlation
# It will have some NaN values as well. It means the corelation doesn't exist. We need to drop them.
corr_starwars.dropna(inplace=True)      # inplace is used to make the changes in same object only
#print(corr_starwars)


# Now, if we have to suggest any movie, we can suggest that movie which has high correlation. But there might be a possibility that a movie 'A' is rated by only 6 persons and 
# each one rated 5. The same persons also rated Star Wars(rated by 583 persons in total) as well. In such case, the corelation will come out to be 1. We should not suggest such movie 'A'.
# So let's put a constraint that the movie to be suggested must have been rated atleast 100 times.

# We can have a dataframe which has correlation value as well as the number of ratings. This data frame can be formed by joining 'corr_starwars' dataframe with 'ratings_df' dataframe
# which has No. of ratings

corr_starwars = corr_starwars.join(ratings_df['No. of ratings'])
#print(corr_starwars)    # Now corr_starwars has three columns: title, Correlation and No. of ratings

suggester = corr_starwars[corr_starwars['No. of ratings']>100].sort_values('Correlation', ascending=False)
#print(suggester)   # Now we can suggest the movies from this data frame from top

##### Predict Function #####

def predict_movies(movie_name):
    movie_user_ratings = moviemat[movie_name]
    similar_movie = moviemat.corrwith(movie_user_ratings)

    corr_movie = pd.DataFrame(similar_movie, columns = ['Correlation'])
    corr_movie.dropna(inplace = True)

    corr_movie = corr_movie.join(ratings_df['No. of ratings'])
    predictions = corr_movie[corr_movie['No. of ratings']>100].sort_values('Correlation', ascending = False)

    return predictions

predictions = predict_movies('Titanic (1997)')
print(predictions.head())