import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 

def pair_generator(df):
    sentence_pair  = []
    sentence_label = []
    for _, row in df.iterrows():
        sentence_pair.append((row['sentence1'], row['sentence2']))
        sentence_label.append(row['gold_label'])
    return sentence_pair, sentence_label


class Vocabulary:
  def __init__(self):
    self.word2index = {}
    self.word2count = {}
    self.index2word = {}
    self.n_words = 0

  def addSentence(self, sentence):
    for word in word_tokenize(sentence):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words + 1
      self.word2count[word] = 1
      self.index2word[self.n_words + 1] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1

# vocab = Vocabulary()

class DataSetLoader(Dataset):
  def __init__(self, sentence_pair, labels):
    self.sentence_pair = sentence_pair
    self.labels        = labels

  def __len__(self):
    return len(self.sentence_pair)

  def __getitem__(self, index):
    return self.sentence_pair[index], self.labels[index]

def get_pair_indices(vocab, sentence_pairs):
  indices_pairs = []
  for sentence_pair in sentence_pairs:
    premise = sentence_pair[0]
    premise_indices = [vocab.word2index[w] for w in word_tokenize(premise)]
    hypothesis = sentence_pair[1]
    hypothesis_indices = [vocab.word2index[w] for w in word_tokenize(hypothesis)]
    indices_pairs.append((premise_indices, hypothesis_indices))
  return indices_pairs