# Copyright 2018 Christopher John Bayron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import chain
import math
import configparser
import os
from collections import Counter
import pickle

import numpy as np

from crawler import db_operations
from embeddings import word2vec
from common import utils as ut
from common import constants as const

# valid punctuations (to keep)
valid_punc_list = [',','?','!','...','.']
# invalid punctuations (to remove)
invalid_punc_list = ['"','(',')']
# words/symbols to replace
replace_word_dict = {
    '..':'.',
    '–':'-',
    '~':' '
    }

def dump_raw_lyrics(all_lyrics):
    """ Dump the raw lyrics to a file
    """
    raw_lyrics_path = 'logs/raw_lyrics.txt'
    with open(raw_lyrics_path, 'w') as f:
        for lyrics in all_lyrics:
            f.write(lyrics[0])

    print("Saved raw lyrics to %s." % raw_lyrics_path)

def dump_processed_lyrics(chained_lyrics):
    """ Dump the processed lyrics to a file
    """
    proc_lyrics_path = 'logs/processed_lyrics.txt'
    with open(proc_lyrics_path, 'w') as f:
        f.write(chained_lyrics)

    print("Saved processed lyrics to %s." % proc_lyrics_path)

def preprocess_lyrics(all_lyrics):
    """ Data cleaning
    """
    lyrics_seq_list = []
    for idx, lyrics in enumerate(all_lyrics):
        # \n\n\n\n - stanza separator
        # \r\n - song start
        # \n\n - end of line
        # remaining \n - end of song
        lyrics = lyrics[0].replace('\n\n\n\n',' . ') \
                          .replace('\r\n','') \
                          .replace('\n\n',' ') \
                          .replace('\n',' . ').lower()
        if idx == 0:
            lyrics = '. ' + lyrics
        # add space to desired punctuation marks to treat them as words                       
        for p in valid_punc_list:
            lyrics = lyrics.replace(p, ' ' + p)
        # remove undesired symbols
        for p in invalid_punc_list:
            lyrics = lyrics.replace(p, '')
        # replace some words/symbols
        while any(w in lyrics for w in replace_word_dict):
            for word in replace_word_dict:
                lyrics = lyrics.replace(word, replace_word_dict[word])
        # replace consecutive periods with single period
        while '. .' in lyrics:
            lyrics = lyrics.replace('. .', '.')
        # tokenize words
        lyrics_words = lyrics.split(' ')
        # remove empty strings
        lyrics_words = list(filter(None, lyrics_words))
        lyrics_seq_list.append(lyrics_words)

    return lyrics_seq_list

def filter_by_freq(word_list, chained_words, min_freq):
    """ Replaces words occuring less than min_freq with the unknown tag <unk>
    and dumps the replaced words to a file (via pickle)
    """
    unk_list = []
    print("Filtering words occuring less than %d times..." % min_freq)
    for word in word_list:
        if word_list.count(word) < min_freq:
            chained_words = chained_words.replace(' ' + word + ' ', ' <unk> ')
            unk_list.append(word)

    print("Filtered %d words." % len(unk_list))
    if len(unk_list) > 0:
        unk_list_path = 'embeddings/unk_list.pkl'
        with open(unk_list_path, 'wb') as ufp:
            pickle.dump(unk_list, ufp)
        print("Saved filtered words in %s." % unk_list_path)
    
    word_list = chained_words.split(' ')
    return chained_words, word_list

def dump_word_freq(lyrics_list):
    """ Prints the number of occurence of each word in 
    lyrics_list to a file
    """
    word_freq_fp = 'logs/word_freq.txt'
    counter_items = Counter(lyrics_list).most_common()
    with open(word_freq_fp, 'w') as f:
        for word, cnt in counter_items:
            print(word, cnt, file=f)
        print("Saved word frequencies in %s." % word_freq_fp)

if __name__ == '__main__':
    
    # read config
    config = configparser.ConfigParser()
    config.read(const.CONFIG_NAME)

    bptt = int(config['training']['bptt'])
    batch_size = int(config['training']['batch_size'])
    # config for optional steps
    show_tsne = bool(int(config['embeddings']['show_tsne']))
    dump_raw = bool(int(config['embeddings']['dump_raw_lyrics']))
    dump_processed = bool(int(config['embeddings']['dump_processed_lyrics']))
    dump_freq = bool(int(config['embeddings']['dump_word_freq']))
    # path to save TF Records
    rec_dir = config['training']['tf_rec_dir']
    rec_fn = config['training']['tf_rec_fn']
    rec_path = os.path.join(rec_dir, rec_fn)
    # get artists
    artists_csv = config['crawler']['artists']
    artists_list = list(filter(None, artists_csv.split(',')))
    # load lyrics
    load_lyrics = bool(int(config['embeddings']['load_lyrics']))
    load_lyrics_path = config['embeddings']['load_lyrics_path']
    # minimum word frequency
    min_word_freq = int(config['embeddings']['min_word_freq'])

    if load_lyrics is True:
        # load lyrics from a file
        # this assumes that the lyrics in the file are already pre-processed
        print("Loading lyrics from %s..." % load_lyrics_path)
        with open(load_lyrics_path, 'r') as lyrics_file:
            chained_lyrics = lyrics_file.read()
    else:
        # fetch all lyrics from DB
        all_lyrics = db_operations.fetch_all_lyrics_from_artists(artists_list)
        if dump_raw is True:
            dump_raw_lyrics(all_lyrics)
        # perform some preprocessing
        lyrics_seq_list = preprocess_lyrics(all_lyrics)
        # NOW lyrics_seq_list is a list of sequences, where each sequence is a list of words

        # lyrics_list is a LIST concatenation of all the sequences
        lyrics_list = list(chain.from_iterable(lyrics_seq_list))
        # chained_lyrics is a STRING concatenation of all the sequences
        chained_lyrics = ' '.join(lyrics_list)

    while '. .' in chained_lyrics:
        chained_lyrics = chained_lyrics.replace('. .', '.')

    lyrics_list = chained_lyrics.split(' ')
    if dump_freq:
        dump_word_freq(lyrics_list)

    if min_word_freq > 1:
        chained_lyrics, lyrics_list = \
            filter_by_freq(lyrics_list, chained_lyrics, min_word_freq)

    if dump_processed is True:
        dump_processed_lyrics(chained_lyrics)

    # from the lyrics, create the inputs and labels(next words) for our model
    inputs = np.array(lyrics_list[:-1])
    labels = np.array(lyrics_list[1:])
    # divide data according to batch size and BPTT
    time_batched_inputs = ut.time_batch_split(inputs, batch_size, bptt)
    time_batched_labels = ut.time_batch_split(labels, batch_size, bptt)
    # generate embeddings
    model, word_to_emb = \
        word2vec.make_wordvectors([lyrics_list], config)
    # replace all words in input with embeddings
    input_emb = ut.word_list_to_emb_list(time_batched_inputs, word_to_emb)
    label_idx = ut.word_list_to_idx_list(time_batched_labels, model.wv.vocab)
    # save inputs, labels as TF Records
    ut.save_seq2seq_as_TFRecord(rec_path, input_emb, label_idx)
    print("Saved %d embedding-label pairs to %s!" %
        (len(input_emb), rec_path))
    print("Vocab Size: %d" % len(model.wv.vocab))
    # save counts to config
    config['embeddings']['num_seq'] = str(len(input_emb))
    config['embeddings']['vocab_size'] = str(len(model.wv.vocab))
    with open(const.CONFIG_NAME, 'w') as config_file:
        config.write(config_file)
    # plot the generated embeddings
    if show_tsne is True:
        ut.tsne_plot(model)
