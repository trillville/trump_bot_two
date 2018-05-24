import re
import json
import sys
import numpy as np
import twitter
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
def generate_output_tweet(seed_tweet):
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

def main():
    # Load character dictionaries
    char_indices = json.load(open('char_indices.json'))
    indices_char = json.load(open('indices_char.json'))

    # Load model
    model = load_model('model.h5')

    # Static tweet for testing
    tweet_text = "California finally deserves a great Governor one who understands borders crime and lowering taxes. John Cox is the man - he’ll be the best Governor you’ve ever had. I fully endorse John Cox for Governor and look forward to working with him to Make California Great Again!"

    api = twitter.Api(consumer_key='biZ0AS0JnEvMaG4iQFmtT78bZ',
                      consumer_secret='1znTvAD8baeKp0escN37ZnxGUqudxA1ukwZSz9mBAbh9io6wPd',
                      access_token_key='999491256307728390-ELrLWW8Juhz0V7yZfgXUPI9CLSVLmoq',
                      access_token_secret='ppRXEnpP01kg1unqIdAeJZjT9tEZKlwgSmJJny2iW3x88')

    # Generate tweet
    tweet_formatted = preprocess_input_tweet(tweet_text)
    generate_output_tweet(tweet_formatted)

    # Generate tweet
    tweet_formatted = preprocess_input_tweet(tweet_text)
    output_tweet = generate_output_tweet(tweet_formatted)
    # Post tweet
    api.PostUpdate(output_tweet)

if __name__ == "__main__":
    #main(sys.argv[1])
    main()
