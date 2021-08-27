import os
import pandas as pd
import tqdm
import re
import torch
import json
# import torch_xla
# import torch_xla.core.xla_model as xm
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
# from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import tensorflow as tf
from preprocess import *
from vocabulary import *
from model import *
from model_functions import *

import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from string import punctuation
import re
from gensim.models import KeyedVectors
import numpy as np


print("\nReading train, test df.......")
train_df = pd.read_csv('../../../evaluation/snli/snli_/snli_1.0_train.csv',  encoding='utf-8')
val_df   = pd.read_csv('../../../evaluation/snli/snli_/snli_1.0_dev.csv',  encoding='utf-8')
# train_df = pd.read_csv('../datasets/snli/snli_/snli_1.0_train.csv')
# val_df   = pd.read_csv('../datasets/snli/snli_/snli_1.0_dev.csv')  <-- personalized read directories
train_df, val_df = preprocess_text(train_df, val_df)
# print(len(train_df))
# print(len(val_df))

print("\nReading custom df1.......")
# rw = pd.read_csv('../datasets/Transformer_generated_summaries.csv', encoding='utf-8')
# rw = pd.read_csv('ins.csv', encoding='utf-8')
rw = pd.read_csv('test_sets/based_ins.csv', encoding='utf-8')
# rw = pd.read_csv('../datasets/test_data/Transformer_generated_summaries.csv', encoding='utf-8')
rw.drop(['original_summary'], axis = 1, inplace=True)
rw.rename(columns = {'text':'sentence1'}, inplace = True)
rw.rename(columns = {'predicted_summary':'sentence2'}, inplace = True)
# rw.rename(columns = {'Predicted_summary':'sentence2'}, inplace = True)
# print(rw)
gold = [-1] * 100
rw.insert(loc=0, column='gold_label', value=gold)
rw = rw[['gold_label', 'sentence1', 'sentence2']]
# rw = rw[['sentence1', 'sentence2']]
rw = rw.dropna()
# rw.head()

print("\nReading custom df2.......")
# rw = pd.read_csv('../datasets/Transformer_generated_summaries.csv', encoding='utf-8')
# rw = pd.read_csv('ins.csv', encoding='utf-8')
# rw_ = pd.read_csv('test_sets/ins.csv', encoding='ansi')
rw_ = pd.read_csv('test_sets/based_ins.csv', encoding='utf-8')
# rw = pd.read_csv('../datasets/test_data/Transformer_generated_summaries.csv', encoding='utf-8')
rw_.drop(['text'], axis = 1, inplace=True)
rw_.rename(columns = {'original_summary':'sentence1'}, inplace = True)
rw_.rename(columns = {'predicted_summary':'sentence2'}, inplace = True)
# rw_.rename(columns = {'Predicted_summary':'sentence2'}, inplace = True)
gold = [-1] * 100
rw_.insert(loc=0, column='gold_label', value=gold)
rw_ = rw_[['gold_label', 'sentence1', 'sentence2']]
# rw = rw[['sentence1', 'sentence2']]
rw_ = rw_.dropna()
# rw_.head()

print("\nClean Texting train and test sets, and rw and rw_.......")
train_df['sentence1'] = train_df['sentence1'].astype(str).apply(lambda text: clean_text(text))
train_df['sentence2'] = train_df['sentence2'].astype(str).apply(lambda text: clean_text(text))
val_df['sentence1'] = val_df['sentence1'].astype(str).apply(lambda text: clean_text(text))
val_df['sentence2'] = val_df['sentence2'].astype(str).apply(lambda text: clean_text(text))
rw['sentence1'] = rw['sentence1'].astype(str).apply(lambda text: clean_text(text))
rw['sentence2'] = rw['sentence2'].astype(str).apply(lambda text: clean_text(text))
rw_['sentence1'] = rw_['sentence1'].astype(str).apply(lambda text: clean_text(text))
rw_['sentence2'] = rw_['sentence2'].astype(str).apply(lambda text: clean_text(text))

print("\nPreProcessing train and test sets.......")
train_df = train_df[(train_df['sentence1'].str.split().str.len() > 0) & (train_df['sentence2'].str.split().str.len() > 0)]
val_df = val_df[(val_df['sentence1'].str.split().str.len() > 0) & (val_df['sentence2'].str.split().str.len() > 0)]
print(train_df[(train_df['sentence1'].str.split().str.len() == 0) | (train_df['sentence2'].str.split().str.len() == 0)])
print(val_df[(val_df['sentence1'].str.split().str.len() == 0) | (val_df['sentence2'].str.split().str.len() == 0)])

train_val_df = pd.concat([train_df, val_df])

print("\ngenerating sentence pairs.......")
sentence_pairs, _ = pair_generator(train_val_df)
rw_sentence_pairs, __ = pair_generator(rw)
rw_sentence_pairs_, __ = pair_generator(rw_)
train_sentence_pairs, train_sentence_labels = pair_generator(train_df)
val_sentence_pairs, val_sentence_labels = pair_generator(val_df)

print("\ngenerating entailment labels.......")
labels = set(train_sentence_labels)
print(labels)
# tag2idx = {word: i for i, word in enumerate(labels)}
# print(tag2idx)
tag2idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
print(tag2idx)
train_labels = [tag2idx[t] for t in train_sentence_labels]
val_labels = [tag2idx[t] for t in val_sentence_labels]

# Adding to Vocabulary
print("\nAdding to custom Vocabulary.......")
vocab = Vocabulary()
for data in [rw_sentence_pairs]:
  for sen in data:
    premise    = sen[0]
    hypothesis = sen[1]
    vocab.addSentence(premise)
    vocab.addSentence(hypothesis)
for data in [rw_sentence_pairs_]:
  for sen in data:
    premise    = sen[0]
    hypothesis = sen[1]
    vocab.addSentence(premise)
    vocab.addSentence(hypothesis)
for data in [sentence_pairs]:
  for sentence_pair in data:
    premise    = sentence_pair[0]
    hypothesis = sentence_pair[1]
    vocab.addSentence(premise)
    vocab.addSentence(hypothesis)

with open('saved_vocabs/based_voc.class', 'wb') as vocab_file:
    pickle.dump(vocab, vocab_file)
print("Vocab size:", len(vocab.word2index))

print('\ndone\n')