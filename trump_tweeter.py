import json
import sys
import numpy as np
import twitter
import os
import psycopg2
import random
from keras.models import load_model
from utils import get_time, id_or_none, preprocess_input_tweet, clean_output_tweet, sample

NUM_CHARS = 37
CHARACTER_LIMIT = 279
MODEL_INPUT_LEN = 130
MIN_TEMP = 0.20
MAX_TEMP = 0.40
WORD_TEMP = 0.30
BASE_SEED_TWEET = '~While in the Philippines I was forced to watch @CNN which I have not done in months and again realized how bad and FAKE it is. Loser!`'

# What is trump gonna say next!
def generate_output_tweet(model, seed_tweet, char_indices, indices_char):
    sentence_trunc = seed_tweet[-MODEL_INPUT_LEN:]
    generated, next_char = '', ''
    diversity = MAX_TEMP
    word_index = 0
    while (next_char != '`' or len(generated) < 2) and len(generated) < 500:
        x_pred = np.zeros((1, MODEL_INPUT_LEN, NUM_CHARS))
        for t, char in enumerate(sentence_trunc):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[str(next_index)]
        if next_char == ' ':
            diversity = WORD_TEMP
        else:
            diversity = max(MIN_TEMP, diversity - 0.025)
        generated += next_char
        sentence_trunc = sentence_trunc[1:] + next_char

    # Clean up output text a bit
    clean = clean_output_tweet(generated)
    if len(clean) >= CHARACTER_LIMIT:
        return split_tweet(clean)
    else:
        return [clean]

# Split tweets longer than 280 characters into smaller tweets, separate with ...
def split_tweet(tweet_text):
    out = []
    while len(tweet_text) >= (CHARACTER_LIMIT - 3):
        out.append(tweet_text[0:CHARACTER_LIMIT - 3] + '...')
        tweet_text = tweet_text[CHARACTER_LIMIT - 3:]
    out.append("..." + tweet_text)
    return out

# Check for new tweets!
def get_new_tweets(cur, api):
    cur.execute("CREATE TABLE IF NOT EXISTS last_tweet_db (tweet_id BIGINT)")
    cur.execute("SELECT max(tweet_id) as tweet_id FROM last_tweet_db")
    last_tweet = id_or_none(cur.fetchone())
    api_tweets = api.GetUserTimeline(screen_name='realDonaldTrump', since_id=last_tweet)
    if len(api_tweets) == 0:
        return None
    else:
        cur.execute("INSERT INTO last_tweet_db VALUES (%s)", (api_tweets[0].id,))
        all_tweets = []
        for i in range(len(api_tweets)):
            tweet = '~' + preprocess_input_tweet(api_tweets[i].text) + '`'
            # Model needs `MODEL_INPUT_LEN` seed characters to make decent tweets
            if len(tweet) < MODEL_INPUT_LEN:
                tweet = BASE_SEED_TWEET + tweet
            all_tweets.append(tweet[-MODEL_INPUT_LEN:])
        return all_tweets

def main():
    # Connect to Twitter API
    api = twitter.Api(consumer_key=os.environ['CONSUMER_KEY'],
                      consumer_secret=os.environ['CONSUMER_KEY_SECRET'],
                      access_token_key=os.environ['ACCESS_TOKEN'],
                      access_token_secret=os.environ['ACCESS_TOKEN_SECRET'])

    # Connect to Postgres
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    cur = conn.cursor()
    latest_tweets = get_new_tweets(cur, api)

    # Done with Postgres - commit changes to DB and close cursor
    conn.commit()
    cur.close()
    if latest_tweets is None:
        print("NO NEW TWEETS")
        sys.exit(1)

    # These dictionaries map model predictions <-> characters
    char_indices = json.load(open('char_indices_lower.json'))
    indices_char = json.load(open('indices_char_lower.json'))

    # Load model: 2 layer LSTM, 512 units each, dropout = 0.2, RMSprop optimizer
    # trained on semi-redundant (5 char stride) 140 character long tweet chunks
    model = load_model('trump_model_5 (9)')

    # Post tweet(s)
    for input_tweet in latest_tweets:
        print('INPUT: ' + input_tweet)
        output_tweets = generate_output_tweet(model, input_tweet, char_indices, indices_char)
        print(output_tweets)
        for tweet in reversed(output_tweets):
            api.PostUpdate(tweet)

if __name__ == '__main__':
    main()
