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
MODEL_INPUT_LEN = 140
CHARACTER_LIMIT = 279
POSSIBLE_HOURS = ['7','8','9']
POSSIBLE_MINUTES = ['00','15','30','45']
MIN_TEMP = 0.20
MAX_TEMP = 0.40
BASE_SEED_TWEET = '~While in the Philippines I was forced to watch @CNN which I have not done in months and again realized how bad and FAKE it is. Loser!`'

# Clean up input tweet
def preprocess_input_tweet(tweet_text):
    t1 = re.sub(r'[^A-Za-z@!.?~`:/ ]', '', tweet_text)
    t2 = re.sub(r'(?<=[!])(?=[^\s])', r' ', t1)
    t3 = re.sub(r' !', '', t2)
    t4 = re.sub(' +',' ', t3)
    out = re.sub(r' @ ', '', t4).replace('\n', '').replace('&amp;', '')
    return out

# Clean up output tweet (have to break URLs so twitter API doesnt complain about fake links)
def clean_output_tweet(raw_text):
    return re.sub('`|~', '', raw_text).replace('://', ':/') \
                                       .replace('amp', '&') \
                                       .replace(' : ', get_time(long=True)) \
                                       .replace('at pm', 'at' + get_time() + 'pm') \
                                       .replace('at am', 'at' + get_time() + 'am')

# Sample a character, with a given temperature (diversity) parameter
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# What is trump gonna say next!
def generate_output_tweet(model, seed_tweet, char_indices, indices_char):
    sentence_trunc = seed_tweet[-MODEL_INPUT_LEN:]
    generated, next_char = '', ''
    diversity = random.uniform(MIN_TEMP, MAX_TEMP)
    while (next_char != '`' or len(generated) < 2) and len(generated) < 500:
        x_pred = np.zeros((1, MODEL_INPUT_LEN, NUM_CHARS))
        for t, char in enumerate(sentence_trunc):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[str(next_index)]
        generated += next_char
        sentence_trunc = sentence_trunc[1:] + next_char

    # Clean up output text a bit
    clean = clean_output_tweet(generated)
    if len(clean) >= CHARACTER_LIMIT:
        return split_tweet(clean)
    else:
        return [clean]

# I removed numbers as possible characters to simplify model - add them back in!
def get_time(long=False):
    if long is False:
        return ' ' + random.choice(POSSIBLE_HOURS)
    else:
        return ' ' + random.choice(POSSIBLE_HOURS) + ':' + random.choice(POSSIBLE_MINUTES) + ' '

def id_or_none(tweet_tuple):
    if tweet_tuple[0] is None:
        return None
    else:
        return int(tweet_tuple[0])

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
                      access_token_secret=os.environ['ACCESS_TOKEN_SECRET']))

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
    char_indices = json.load(open('char_indices.json'))
    indices_char = json.load(open('indices_char.json'))

    # Load model: 2 layer LSTM, 512 units each, dropout = 0.2, RMSprop optimizer
    # trained on semi-redundant (5 char stride) 140 character long tweet chunks
    model = load_model('model.h5')

    # Post tweet(s)
    for input_tweet in latest_tweets:
        print('INPUT: ' + input_tweet)
        output_tweets = generate_output_tweet(model, input_tweet, char_indices, indices_char)
        print(output_tweets)
        for tweet in output_tweets:
            api.PostUpdate(tweet)

if __name__ == '__main__':
    main()
