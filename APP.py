#MOVIE RECOMMENDER SYSTEM BASED ON COSINE SIMILARITY AND WEIGHTED AVERAGES
#DATA PREPROCESSING
#Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Load the datasets
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

#Show credits datasets
credits.head()

#Show movies dataset
movies.head()

credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_merge = movies.merge(credits_column_renamed, on='id')
movies_merge.head()

movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status','production_countries'])
movies_cleaned.head()

movies_cleaned.info()

# Calculate all the components based on the weighted averagr formula
v=movies_cleaned['vote_count']
R=movies_cleaned['vote_average']
C=movies_cleaned['vote_average'].mean()
m=movies_cleaned['vote_count'].quantile(0.70)
movies_cleaned['weighted_average']=((R*v)+ (C*m))/(v+m)

#Add weighted average column to the previous dataset
movies_cleaned['weighted_average']=((R*v)+ (C*m))/(v+m)
movies_cleaned.head()

print("New Movies Dataframe:",movies_cleaned.shape)

#Ranking the dataset according to the weighted average
movie_ranking=movies_cleaned.sort_values('weighted_average',ascending=False)
movie_ranking[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(20)

import matplotlib.pyplot as plt
import seaborn as sns

weight_average=movie_ranking.sort_values('weighted_average',ascending=False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=weight_average['weighted_average'].head(10), y=weight_average['original_title'].head(10), data=weight_average)
plt.xlim(4, 10)
plt.title('Best Movies by average votes', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')
plt.savefig('best_movies.png')


#Visualize Vote average vs popularity
plt.figure(figsize=(12, 6))
sns.scatterplot(
    x=movies_cleaned['popularity'],
    y=movies_cleaned['vote_average'],
    
    
)
plt.title('Vote Average VS Popularity', weight='bold')
plt.xlabel('Popularity', weight='bold')
plt.ylabel('Vote Average', weight='bold')
plt.legend(title='Cluster')
plt.savefig('movies_clusters.png')
plt.show()

#Implementing the recommender system using cosine similarity and weighted means.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Prepare the Feature (Overview)
movies_cleaned['overview'] = movies_cleaned['overview'].fillna('')  # Replace NaN overviews with empty strings

#Convert Text to Numerical Vectors (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_cleaned['overview'])

#Compute Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Normalize Both Scores (Cosine Similarity and Weighted Average)
cosine_sim_normalized = cosine_sim.max(axis=1) - cosine_sim  # Normalizing Cosine Similarity (inverse distance)
cosine_sim_normalized = 1 - cosine_sim_normalized / cosine_sim_normalized.max(axis=1)[:, np.newaxis]  # Normalize to [0, 1]

#Normalize Weighted Average (scale between 0 and 1)
weighted_average_normalized = (movies_cleaned['weighted_average'] - movies_cleaned['weighted_average'].min()) / \
                              (movies_cleaned['weighted_average'].max() - movies_cleaned['weighted_average'].min())

#Combine the Scores
alpha = 0.7  # Adjust the weight for cosine similarity
beta = 1 - alpha  # Adjust the weight for weighted average
combined_scores = alpha * cosine_sim_normalized + beta * weighted_average_normalized.values[:, np.newaxis]

#Function to Recommend Movies Based on Combined Score
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()

def recommend_movies(title, cosine_sim=cosine_sim, combined_scores=combined_scores, top_n=10):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the combined scores for all movies with the given movie
    sim_scores = list(enumerate(combined_scores[idx]))

    # Sort the movies based on combined scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top_n most similar movies (excluding itself)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    # Return the top_n most similar movies
    return movies_cleaned.iloc[movie_indices][['original_title', 'weighted_average', 'popularity']]

# Step 8: Test the Combined Recommendation System
movie_title = "Spy Kids"  # Change to any movie title in the dataset
recommended_movies = recommend_movies(movie_title)
print(f"Movies recommended based on '{movie_title}':\n", recommended_movies)

import streamlit as st
import pandas as pd

# Extract the movie names into a list
movie_list = movies['original_title'].tolist()

# Convert all movie names to lowercase for case-insensitive comparison
movie_list_lower = [movie.lower() for movie in movie_list]

# Add background image (optional, can be customized)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://media.istockphoto.com/id/1302499179/vector/realistic-3d-film-strip-cinema-on-blue-background-with-place-for-text-modern-3d-isometric.jpg?s=612x612&w=0&k=20&c=ekfhQYcRwVnl-yHeieIHTLXehHUL2bD6ioaGn7Q3Nf4=');
         background-size: cover;
        background-attachment: fixed;
        color: white;  /* Ensures text is readable over the background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit title
st.title("Movie Recommender System")

# Display a button for the recommender
if st.button("Show Movie List"):
    # Display the movie list below the button in a clearer format
    st.write("Movies in the database:")
    movie_df = pd.DataFrame(movie_list, columns=["Movie Titles"])
    movie_df.index += 1  # Adjust the index to start from 1
    
    # Display the DataFrame
    st.dataframe(movie_df)
# Input box for the user to type a movie name
movie_name = st.text_input("Enter a movie name:")

# Check if the user entered a movie name
if movie_name:
    # Convert the user input to lowercase for case-insensitive search
    movie_name_lower = movie_name.lower()

    # Check if the movie is in the list
    if movie_name_lower in movie_list_lower:
        # Find the exact movie name (case-sensitive) in the list
        movie_index = movie_list_lower.index(movie_name_lower)
        st.write(f"You searched for: {movie_list[movie_index]}")
        
        # Show recommended movies separately (you can use your own recommendation logic here)
        st.subheader("Recommended Movies:")
        # Example of displaying a few movie recommendations (you can replace this with your logic)
        for i in range(movie_index + 1, movie_index + 6):
            if i < len(movie_list):
                st.write(movie_list[i])
    else:
        st.write("Sorry, the movie is not found in the database. Please try again.")
else:
    st.write("Please type a movie name to get started.")


