import re
import json
import sys
import numpy as np
import twitter
import os
import psycopg2
from keras.models import load_model

NUM_CHARS = 61
MAX_LEN = 140

# Hacky regex... but cleans up the text a little bit
def preprocess_input_tweet(tweet_text):
    t1 = re.sub(r'[^A-Za-z@!.?~`:/ ]', '', tweet_text)
    t2 = re.sub(r'(?<=[!])(?=[^\s])', r' ', t1)
    t3 = re.sub(r' !', '', t2)
    t4 = re.sub(' +',' ', t3)
    out = re.sub(r' @ ', '', t4).replace('\n', '').replace('&amp;', '') + '`'
    return out[-MAX_LEN:]

# Sample a character, with a given temperature/diversity parameter
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# What is trump gonna say next!
def generate_output_tweet(seed_tweet, char_indices, indices_char):
    sentence_trunc = seed_tweet[-MAX_LEN:]
    generated = ''
    next_char = ''
    diversity = 0.2
    print(sentence_trunc)
    while next_char != '`' or len(generated) < 2:
        x_pred = np.zeros((1, MAX_LEN, NUM_CHARS))
        for t, char in enumerate(sentence_trunc):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[str(next_index)]

        generated += next_char
        sentence_trunc = sentence_trunc[1:] + next_char
    return(generated)

def get_new_tweets(conn, api):
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS last_tweet_db (tweet_id BIGINT);")
    cur.execute("SELECT tweet_id FROM last_tweet_db")
    last_tweet = cur.fetchone()
    api_tweets = api.GetUserTimeline(screen_name="realDonaldTrump")
    if last_tweet is None or last_tweet != api_tweets[0].id:
        cur.execute("INSERT INTO last_tweet_db VALUES (%s)", (api_tweets[0].id,))
        input_tweet = api_tweets[0].text
        i = 1
        while len(input_tweet) < 200:
            input_tweet = api_tweets[i].text + input_tweet
            i += 1
        return input_tweet
    else:
        print("NO NEW TWEETS")
        sys.exit(1)

def main():
    # Connect to Twitter API
    api = twitter.Api(consumer_key=os.environ.get('CONSUMER_KEY'),
                      consumer_secret=os.environ.get('CONSUMER_KEY_SECRET'),
                      access_token_key=os.environ.get('ACCESS_TOKEN'),
                      access_token_secret=os.environ.get('ACCESS_TOKEN_SECRET'))

    # Connect to Postgres
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    latest_tweet = get_new_tweets(conn, api)

    # Load character dictionaries
    char_indices = json.load(open('char_indices.json'))
    indices_char = json.load(open('indices_char.json'))

    # Load model
    model = load_model('model.h5')

    # Generate tweet
    tweet_formatted = preprocess_input_tweet(latest_tweet)
    output_tweet = generate_output_tweet(tweet_formatted, char_indices, indices_char)

    # Post tweet
    api.PostUpdate(output_tweet)

if __name__ == "__main__":
    #main(sys.argv[1])
    main()
