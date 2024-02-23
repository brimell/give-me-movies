# Movie Recommendation System Project

## Overview

This project aims to create a personalized movie recommendation system. It leverages user ratings data to match users with similar tastes and recommend movies that they might enjoy based on their shared preferences. The recommendation logic is built on the principle of collaborative filtering, focusing on finding users with similar rating patterns and suggesting movies highly rated by these like-minded users but not yet watched by the target user.

## Data Description

The project utilizes three main datasets:

1. **User Ratings (`user_ratings.csv`)**: This dataset contains users' ratings for various movies. Each row represents a rating given by a user to a movie, including the user ID, movie ID, and the rating value.
2. **Movies (`movies.csv`)**: This dataset provides detailed information about each movie, including its title, genres, release year, IMDb and TMDB links, popularity, vote count, and more.
3. **My Ratings (`my_ratings.csv`)**: A personal dataset containing the movies you have watched and rated. It includes columns for the date you watched the movie, the movie's name, the year it was released, its Letterboxd URI, and your rating.

## System Dependencies

The project requires Python 3.x and the following Python libraries:

-   pandas
-   numpy
-   scipy
-   sklearn

## Setup and Installation

1. **Clone the Repository**: Clone this repository to your local machine to get started with the project.

2. **Install Dependencies**: Install the required Python libraries using pip:

```shell
pip install pandas numpy scipy scikit-learn
```

3. **Prepare the Data**: Place your `user_ratings.csv`, `movies.csv`, and `my_ratings.csv` files in the project directory.

4. **Run the Script**: Execute the main script to generate movie recommendations. Make sure to adjust paths in the script if your data files are located in a different directory.

## Usage

The main script of the project follows these steps:

1. Load the datasets.
2. Merge the user ratings with the movies dataset to enrich the ratings data with movie information.
3. Process your personal ratings and integrate them with the user ratings data.
4. Calculate similarity scores between you and other users based on your mutual ratings.
5. Identify users with similar tastes.
6. Recommend movies watched and highly rated by these users but not yet rated by you.

## Recommendations

The system outputs a list of recommended movies based on the ratings of users with similar tastes. These recommendations are sorted by their popularity or vote count to ensure that highly regarded movies are suggested.

## Visualization

To help you understand the recommendation process, the project includes a visualization step that displays the relationships between users and their movie ratings, highlighting the common preferences that lead to recommendations.