import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time
from datetime import datetime


# Load API keys
load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")


def fetch_mixed_tweets(max_results=20):
    """
    Fetch a mix of bullying and non-bullying tweets in one request.
    Skips tweets with links or single-word tweets.
    Only English tweets are kept.
    """
    if not BEARER_TOKEN:
        raise ValueError("Twitter Bearer Token not found in .env file")


    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}


    # Combine bullying + non-bullying topics using OR
    query = "(#cyberbullying OR bully OR harass OR abuse OR rude) OR (#motivation OR #movies OR #technology) lang:en"


    params = {
        "query": query,
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,author_id,text,lang"
    }


    response = requests.get(url, headers=headers, params=params)


    # Handle rate limit
    if response.status_code == 429:
        print("⏳ Rate limit reached. Waiting 15 minutes...")
        time.sleep(15 * 60)
        return fetch_mixed_tweets(max_results)


    # Handle other errors
    if response.status_code != 200:
        print(f"⚠️ API Error: {response.status_code}")
        print(response.text)
        return pd.DataFrame(columns=["platform", "text", "label", "fetched_at"])


    data = response.json().get("data", [])
    tweets = []


    bullying_keywords = ["bully", "harass", "abuse", "cyberbullying", "rude"]


    for t in data:
        text = t["text"].strip()


        # Skip tweets with links or only 1 word
        if "http" in text or len(text.split()) <= 1:
            continue


        label = 1 if any(word in text.lower() for word in bullying_keywords) else 0
        tweets.append({
            "platform": "Twitter",
            "text": text,
            "label": label,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })


    df = pd.DataFrame(tweets)
    return df




if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    csv_path = "data/twitter_dataset.csv"


    df_new = fetch_mixed_tweets(max_results=20)


    if not df_new.empty:
        if os.path.exists(csv_path):
            # Load existing data
            df_old = pd.read_csv(csv_path)
            # Combine and remove duplicates based on 'text'
            df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset="text")
        else:
            df_combined = df_new


        df_combined.to_csv(csv_path, index=False)
        print(f"✅ Saved {len(df_new)} new tweets, total now: {len(df_combined)} in data/twitter_dataset.csv")
        print(df_new.head())
    else:
        print("❌ No tweets fetched. Try again later or check API key.")





