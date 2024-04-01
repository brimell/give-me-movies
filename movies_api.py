import requests
import json
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_key = os.getenv("API_KEY")

# Read in data
df = pd.read_csv("data/watched.csv")

# For Update Run: Change date to last script execution
df = df[(df["Date"] > "2023-11-17")]


# Drop Letterboxd URI
df.drop(["Letterboxd URI"], axis=1, inplace=True)
# Initialize columns for desired info
df["id"] = 000000
df["original_language"] = "en"
df["overview"] = "blank"
df["popularity"] = 0.00
df["vote_average"] = 0.00
df["vote_count"] = 0.00
df["genres"] = "blank"
df["revenue"] = 000000
df["runtime"] = 000
df["tagline"] = "blank"


# Initial for loop to pull high-level info about the film
for i in range(0, len(df)):
    title = format(df.iloc[i, 1])
    query = (
        "https://api.themoviedb.org/3/search/movie?api_key="
        + API_key
        + "&query="
        + title
        + ""
    )
    response = requests.get(query)
    if response.status_code == 200:
        json_format = json.loads(response.text)
        if len(json_format["results"]) > 0:
            df.iloc[i, 3] = str(json_format["results"][0]["id"])
            df.iloc[i, 4] = str(json_format["results"][0]["original_language"])
            df.iloc[i, 5] = str(json_format["results"][0]["overview"])
            df.iloc[i, 6] = str(json_format["results"][0]["popularity"])
            df.iloc[i, 7] = str(json_format["results"][0]["vote_average"])
            df.iloc[i, 8] = str(json_format["results"][0]["vote_count"])
        else:
            i = i + 1
    else:
        i = i + 1

# Secondary Pull that uses Movie ID to get more info
for i in range(0, len(df)):
    title = format(df.iloc[i, 3])
    query = "https://api.themoviedb.org/3/movie/" + title + "?api_key=" + API_key + ""
    if df.iloc[i, 3] != "000000":
        response = requests.get(query)
        if response.status_code == 200:
            json_format = json.loads(response.text)
            df.iloc[i, 9] = str(json_format["genres"])
            df.iloc[i, 10] = str(json_format["revenue"])
            df.iloc[i, 11] = str(json_format["runtime"])
            df.iloc[i, 12] = str(json_format["tagline"])
        else:
            i = i + 1
    else:
        i = i + 1

print(df)

# Write to CSV if it doesn't exist
if not os.path.exists("data/processed"):
    os.mkdir("data/processed")
if not os.path.exists("data/processed/ratings_tmdb.csv"):
    df.to_csv("data/processed/ratings_tmdb.csv", header=True, index=False)
else:
    # For Update Run:
    df.to_csv("data/processed/ratings_tmdb_newentries.csv", header=True, index=False)
    df_prev = pd.read_csv("data/processed/ratings_tmdb.csv")
    api_results = pd.concat([df_prev, df], ignore_index=True)
    api_results.sort_values(by="Date", inplace=True)
    api_results.to_csv("data/processed/ratings_tmdb.csv", header=True, index=False)
