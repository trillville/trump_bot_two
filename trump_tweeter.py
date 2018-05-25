import re
import json
import sys
import numpy as np
import twitter
import os
import psycopg2
import random
from keras.models import load_model

NUM_CHARS = 61
MAX_LEN = 140
CHARACTER_LIMIT = 279
POSSIBLE_HOURS = ['7','8','9']
POSSIBLE_MINUTES = ['00','15','30','45']
DIVERSITY_RANGE = [0.15,0.35]

# Hacky regex... but cleans up the text a little bit
def preprocess_input_tweet(tweet_text):
    t1 = re.sub(r'[^A-Za-z@!.?~`:/ ]', '', tweet_text)
    t2 = re.sub(r'(?<=[!])(?=[^\s])', r' ', t1)
    t3 = re.sub(r' !', '', t2)
    t4 = re.sub(' +',' ', t3)
    out = re.sub(r' @ ', '', t4).replace('\n', '').replace('&amp;', '')
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
def generate_output_tweet(model, seed_tweet, char_indices, indices_char):
    sentence_trunc = seed_tweet[-MAX_LEN:]
    generated = ''
    next_char = ''
    diversity = random.uniform(DIVERSITY_RANGE[0], DIVERSITY_RANGE[1])
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
    # Get rid of end of tweet/start of tweet char, and break links so twitter doesnt complain
    generated = re.sub('`|~', '', generated).replace('://', ':/') \
                                            .replace(' : ', get_time()) \
                                            .replace('amp', '&')
    if len(generated) >= CHARACTER_LIMIT:
        return split_tweet(generated)
    else:
        return [generated]

# I removed numbers as possible characters to simplify model - add them back in!
def get_time():
    return random.choice(POSSIBLE_HOURS) + ':' + random.choice(POSSIBLE_MINUTES) + ' '

# Split tweets longer than 280 characters into smaller tweets, separate with ...
def split_tweet(tweet_text):
    out = []
    while len(tweet_text) >= (CHARACTER_LIMIT - 3):
        out.append(tweet_text[0:CHARACTER_LIMIT - 3] + '...')
        tweet_text = tweet_text[CHARACTER_LIMIT - 3:]
    out.append("..." + tweet_text)
    return out

# Check for new tweets!
def get_new_tweets(conn, api):
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS last_tweet_db (tweet_id BIGINT)")
    cur.execute("SELECT max(tweet_id) as tweet_id FROM last_tweet_db")
    last_tweet = cur.fetchone()
    api_tweets = api.GetUserTimeline(screen_name="realDonaldTrump")
    if last_tweet[0] is not None:
        if int(last_tweet[0]) == api_tweets[0].id:
            cur.close()
            return None
    cur.execute("INSERT INTO last_tweet_db VALUES (%s)", (api_tweets[0].id,))
    conn.commit()
    cur.close()
    input_tweet = '~' + api_tweets[0].text + '`'
    i = 1
    while len(input_tweet) < 200:
        input_tweet = '~' + api_tweets[i].text + '`' + input_tweet
        i += 1
    return input_tweet

def main():
    # Connect to Twitter API
    api = twitter.Api(consumer_key=os.environ.get('CONSUMER_KEY'),
                      consumer_secret=os.environ.get('CONSUMER_KEY_SECRET'),
                      access_token_key=os.environ.get('ACCESS_TOKEN'),
                      access_token_secret=os.environ.get('ACCESS_TOKEN_SECRET'))

    # Connect to Postgres
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')

    # Get latest tweet(s)
    latest_tweet = get_new_tweets(conn, api)
    if latest_tweet is None:
        print("NO NEW TWEETS")
        sys.exit(1)

    # Load character dictionaries
    char_indices = json.load(open('char_indices.json'))
    indices_char = json.load(open('indices_char.json'))

    # Load model: 2 layer LSTM, 512 units each, dropout = 0.2, RMSprop optimizer
    # Trained on semi-redundant (5 char stride) 140 character long tweet chunks
    model = load_model('model.h5')

    # Generate tweet(s)
    tweet_formatted = preprocess_input_tweet(latest_tweet)
    output_tweets = generate_output_tweet(model, tweet_formatted, char_indices, indices_char)

    # Post tweet(s)
    for tweet in output_tweets:
        api.PostUpdate(tweet)

if __name__ == '__main__':
    main()
