import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

print("Loading datasets...")
ratings = pd.read_csv('data/ratings_export.csv')
movies = pd.read_csv('data/movie_data.csv', on_bad_lines='skip')
my_ratings = pd.read_csv('data/processed/ratings_tmdb_cleaned.csv')

print("Merging datasets...")
user_ratings = pd.merge(ratings, movies, left_on='movie_id', right_on='movie_id')

print("Merging your ratings with movie data...")
merged = my_ratings.merge(movies[['tmdb_id', 'movie_title', 'year_released']], 
                          left_on='id',  
                          right_on='tmdb_id', 
                          how='inner')
merged = merged[merged['Rating'] != 0]
my_ratings_updated = merged[['tmdb_id', 'Rating']].copy()
my_ratings_updated['user_id'] = "brimell"

print("Preparing data for similarity computation...")
tmdb_id_to_idx = {tmdb_id: i for i, tmdb_id in enumerate(user_ratings['tmdb_id'].unique())}
user_id_to_idx = {user_id: i + 1 for i, user_id in enumerate(user_ratings['user_id'].unique())}
user_id_to_idx["brimell"] = 0

combined_ratings = pd.concat([user_ratings, my_ratings_updated.rename(columns={'Rating': 'rating_val'})])
combined_ratings = combined_ratings.dropna(subset=['user_id', 'rating_val'])


# Get the movies rated by you
your_movies = combined_ratings[combined_ratings['user_id'] == "brimell"]

# Merge your_movies with combined_ratings to find common movies
common_movies = pd.merge(your_movies, combined_ratings, on='tmdb_id')

# Group by user_id and count the number of common movies
common_movies_count = common_movies.groupby('user_id_y').size()

# Get the user_ids of users who have rated at least 20 of the same movies as you
filtered_user_ids = common_movies_count[common_movies_count >= 20].index

# Filter combined_ratings for the users who have rated at least 20 of the same movies as you
combined_ratings = combined_ratings[combined_ratings['user_id'].isin(filtered_user_ids)]


print("Creating the sparse matrix...")
rows = combined_ratings['user_id'].map(user_id_to_idx)
cols = combined_ratings['tmdb_id'].map(tmdb_id_to_idx)
data = combined_ratings['rating_val']
ratings_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_id_to_idx), len(tmdb_id_to_idx)))

print("Computing cosine similarity...")
user_similarity = cosine_similarity(ratings_matrix)

print("Finding the most similar users...")
top_similar_users_indices = np.argsort(-user_similarity[0])[1:11]
top_10_user_ids = [list(user_id_to_idx.keys())[list(user_id_to_idx.values()).index(idx)] for idx in top_similar_users_indices]

print("Recommending movies based on the top similar users...")
my_rated_movies = set(my_ratings_updated['tmdb_id'])
recommended_movies_details = {}

for idx in top_similar_users_indices:
    user_id = list(user_id_to_idx.keys())[list(user_id_to_idx.values()).index(idx)]
    user_movies = combined_ratings[combined_ratings['user_id'] == user_id]
    for _, row in user_movies.iterrows():
        tmdb_id = row['tmdb_id']
        if tmdb_id not in my_rated_movies:
            if tmdb_id not in recommended_movies_details:
                recommended_movies_details[tmdb_id] = {'users': [user_id], 'ratings': [row['rating_val']]}
            else:
                recommended_movies_details[tmdb_id]['users'].append(user_id)
                recommended_movies_details[tmdb_id]['ratings'].append(row['rating_val'])

recommended_movies_ids = sorted(recommended_movies_details, key=lambda x: len(recommended_movies_details[x]['users']), reverse=True)[:100]

print("Fetching movie titles and stats...")
recommended_titles_and_stats = []
for tmdb_id in recommended_movies_ids:
    movie_title = movies[movies['tmdb_id'] == tmdb_id]['movie_title'].iloc[0]
    avg_rating = np.mean(recommended_movies_details[tmdb_id]['ratings'])
    num_users = len(recommended_movies_details[tmdb_id]['users'])
    recommended_titles_and_stats.append({
        'title': movie_title,
        'average_rating': avg_rating,
        'recommended_by_users_count': num_users,
    })

print("Outputting recommended movies to file...")
with open('data/recommended_movies.txt', 'w') as f:
    for movie in recommended_titles_and_stats:
        f.write(f"Title: {movie['title']}, Avg Rating: {movie['average_rating']:.2f}, Recommended by {movie['recommended_by_users_count']} Users\n")

print("Done! Recommended movies file has been created.")
