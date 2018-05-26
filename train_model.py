from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM
from keras.optimizers import RMSprop
from utils import preprocess_input_tweet, sample, MODEL_INPUT_LEN
import numpy as np
import random
import sys
import json

BATCH_SIZE = 200
STRIDE = 5
EPOCHS = 20
FILE_NAME = "/Users/tillmanelser/trump_final.txt"

def load_text(filepath, save=False):
    raw = open(filepath).read()
    clean = preprocess_input_tweet(raw)
    chars = sorted(list(set(clean)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    if save is True:
        with open('char_indices.json', 'w') as out:
            json.dump(char_indices, out)
        with open('indices_char.json', 'w') as out:
            json.dump(indices_char, out)
    return clean, chars, char_indices, indices_char

def gen_training_data(text, chars, char_indices):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - MODEL_INPUT_LEN, STRIDE):
        sentences.append(text[i: i + MODEL_INPUT_LEN])
        next_chars.append(text[i + MODEL_INPUT_LEN])

    print('Vectorization...')
    x = np.zeros((len(sentences), MODEL_INPUT_LEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y

def build_model(n_chars, n_units, n_layers, dropout=None, lr=None):
    model = Sequential()
    model.add(LSTM(n_units, return_sequences=True, input_shape=(MODEL_INPUT_LEN, n_chars)))
    if dropout is not None:
        model.add(Dropout(dropout))
    i = 1
    while i < n_layers:
        model.add(LSTM(n_layers))
        if dropout is not None:
            model.add(Dropout(dropout))
        i += 1
    model.add(Dense(n_chars))
    model.add(Activation('softmax'))
    if lr is not None:
        optimizer = RMSprop(lr=lr)
    else:
        optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def on_epoch_end(epoch, logs, save=True):
    # Function invoked at end of each epoch. Prints generated text.
    if save == True:
        print('----- Saving model after Epoch: %d' % epoch)
        model.save('model' % epoch)

    print('----- Generating text after Epoch: %d' % epoch)
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(250):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[str(next_index)]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

def main():
    clean, chars, char_indices, indices_char = load_text(FILE_NAME)
    x, y = gen_training_data(clean, chars, char_indices)
    model = build_model(n_chars=len(chars), n_units=512, n_layers=2, dropout=0.2)

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    model.fit(x, y,
              batch_size=250,
              epochs=EPOCHS,
              callbacks=[print_callback])

if __name__ == '__main__':
    main()
