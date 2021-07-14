import numpy as np

def get_coefs(word, *arr):
  return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
  with open(path, encoding='utf-8') as f:
    return dict(get_coefs(*line.strip().split(' ')) for line in f)