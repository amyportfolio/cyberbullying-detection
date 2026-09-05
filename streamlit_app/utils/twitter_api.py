import os
import random
import requests
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

BEARER_TOKEN = os.getenv("BEARER_TOKEN")

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

LOCAL_DATASET = os.path.join(
    BASE_DIR,
    "data",
    "X_live_demo.csv"
)

DEFAULT_TOPICS = [

    "technology",
    "artificial intelligence",
    "sports",
    "football",
    "cricket",
    "movies",
    "music",
    "education",
    "gaming",
    "health",
    "travel",
    "science"

]


def load_local_dataset(max_results=10):

    try:

        df = pd.read_csv(LOCAL_DATASET)

        if len(df) > max_results:

            df = df.sample(
                n=max_results,
                random_state=random.randint(1, 9999)
            )

        df = df.copy()

        df["platform"] = "Twitter"

        df["topic"] = "Demo Dataset"

        df["Timestamp"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        print("Using Local Demo Dataset")

        return df

    except Exception as e:

        print(e)

        return pd.DataFrame()


def fetch_mixed_tweets(topic=None, max_results=10):

    if topic is None or topic.strip() == "":

        topic = random.choice(DEFAULT_TOPICS)

    if not BEARER_TOKEN:

        print("Bearer Token Missing")

        return load_local_dataset(max_results)

    max_results = max(10, min(max_results, 100))

    url = "https://api.twitter.com/2/tweets/search/recent"

    headers = {

        "Authorization": f"Bearer {BEARER_TOKEN}"

    }

    query = f'"{topic}" lang:en -is:retweet'

    params = {

        "query": query,

        "max_results": max_results,

        "tweet.fields": "created_at,text"

    }

    try:

        response = requests.get(

            url,

            headers=headers,

            params=params

        )

        print("Status :", response.status_code)

        if response.status_code != 200:

            print("Twitter API unavailable.")

            print(response.text)

            return load_local_dataset(max_results)

        data = response.json().get("data", [])

        tweets = []

        for tweet in data:

            tweets.append({

                "platform": "Twitter",

                "topic": topic,

                "text": tweet["text"],

                "Timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            })

        return pd.DataFrame(tweets)

    except Exception as e:

        print(e)

        return load_local_dataset(max_results)