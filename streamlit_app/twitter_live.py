import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

def fetch_mixed_tweets(max_results=10):

    if not BEARER_TOKEN:
        print("⚠️ Twitter Bearer Token not found")
        return pd.DataFrame()

    # Ensure API valid range (10–100)
    max_results = max(10, min(max_results, 100))

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    query = "((cyberbullying OR harass OR pathetic OR idiot OR moron OR fuck) OR (motivation OR movies OR technology)) lang:en -is:retweet"

    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "created_at,text"
    }


    try:
        response = requests.get(url, headers=headers, params=params)

        print("Status:", response.status_code)
        print("Response:", response.text)

        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json().get("data", [])
        tweets = []

        for t in data:
            text = t["text"]
            tweets.append({
                "platform": "Twitter",
                "text": text,
                "label": 0,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        return pd.DataFrame(tweets)

    except Exception as e:
        print("Exception:", e)
        return pd.DataFrame()
