import random
import re
import numpy as np
import nltk

MODEL_INPUT_LEN = 130
POSSIBLE_HOURS = ['7','8','9']
POSSIBLE_MINUTES = ['00','15','30','45']

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
    nltk.download('punkt')
    sub =  re.sub('`|~', '', raw_text).replace('://', ':/') \
                                      .replace(' amp ', ' & ') \
                                      .replace(' : ', get_time(long=True)) \
                                      .replace('at pm', 'at' + get_time() + 'pm') \
                                      .replace('at am', 'at' + get_time() + 'am') \
                                      .replace(' i ', ' I ') \
                                      .replace('u.s.', 'U.S.')
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.tokenize(sub)
    sentences = [sent.capitalize() for sent in sentences]
    return ' '.join(sentences)

# Sample a character, with a given temperature (diversity) parameter
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
