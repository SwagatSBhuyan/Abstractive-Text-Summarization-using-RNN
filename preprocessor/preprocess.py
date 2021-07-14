import os
import re
import pickle
import string
import unicodedata
from random import randint

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from wordcloud import STOPWORDS, WordCloud

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, TimeDistributed

os.system('pip install -q contractions==0.0.48')

from contractions import contractions_dict

# for key, value in list(contractions_dict.items())[:10]:
#     print(f'{key} == {value}')

from pre_func import expand_contractions
# from pre_func import rm_punc_from_word
# from pre_func import rm_punc_from_text
# from pre_func import rm_number_from_text
# from pre_func import rm_stopwords_from_text
from pre_func import clean_text

filename = '../datasets/inshort.xlsx'
df = pd.read_excel(filename).reset_index(drop=True)

df_columns = df.columns.tolist()
df_columns.remove('headlines')
df_columns.remove('text')
df.drop(df_columns, axis='columns', inplace=True)

# Shuffling the df
df = df.sample(frac=1).reset_index(drop=True)

# Converting to lowercase
df.text = df.text.apply(str.lower)
df.headlines = df.headlines.apply(str.lower)

df.headlines = df.headlines.apply(expand_contractions)
df.text = df.text.apply(expand_contractions)
# df.sample(5)

# print(rm_punc_from_word('#cool!'))
# print(rm_punc_from_text("Frankly, my dear, I don't give a damn"))
# print(rm_number_from_text('You are 100times more sexier than me'))
# print(rm_number_from_text('If you taught yes then you are 10 times more delusional than me'))
# print(rm_stopwords_from_text("Love means never having to say you're sorry"))

print(f'Dataset size: {len(df)}')
# df.sample(5)

print(clean_text("Mrs. Robinson, you're trying to seduce me, aren't you?"))

df.text = df.text.apply(clean_text)
df.headlines = df.headlines.apply(clean_text)

# saving the cleaned data
df.to_csv('../datasets/cleaned_data.csv')