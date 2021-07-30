import pandas as pd
from tqdm import tqdm
import concurrent.futures
import nltk
import os
from rouge import Rouge
import time
import sys

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen


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


sys.path.append('evaluation')
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

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen











# review = pd.read_csv('../../datasets/cleaned_data.csv', encoding='utf-8')
# df = review[['headlines','text']]

# os.system('pip install transformers')
# os.system('pip install sentence-transformers')
# os.system('pip install rouge')
# os.system('pip install rouge-score')


def summ(sentence):

    from transformers import pipeline
    summarizer = pipeline("summarization")
    # dicto={'Text':[],'Original_summary':[],'Predicted_summary':[],'bleu':[],'Semantic_similarity':[],'rouge_1_f':[],'rouge_2_f':[],'rouge_l_f':[]}
    # for i in tqdm(range(100)):
    #     dicto['Text'].append(df.text[i])
    #     dicto['Original_summary'].append(df.headlines[i])
    #     dicto['Predicted_summary'].append(summarizer(df.text[i], max_length=20, min_length=5, do_sample=False)[0]['summary_text'])
    # sentence = "uk india business council chairperson said 60 british firms likely increase investment india next years firms positive reforms fdi introduced last two years added india even important britain said"
    return summarizer(sentence, max_length=20, min_length=5, do_sample=False)



def ent(sen1, sen2):

    with open('evaluation/saved_vocabs/custom_voc.class', 'rb') as vocab_file_r:
        vocab = pickle.load(vocab_file_r)
    print("Vocab size:", len(vocab.word2index))
    index2word = {}
    for wrd, idx in vocab.word2index.items():
        # print(wrd, idx)
        index2word[idx] = wrd

    lstm_model = torch.load('evaluation/saved_models/sm_2')
    lstm_model.eval()

    # nnn = int(input("No. of sentence pairs to check for Entailment: "))
    # sentence_o = [''] * nnn
    # sentence_p = [''] * nnn
    # for i in range(nnn):
    #     print("iteration #" + str(i) + ": ")
    #     sentence_o[i] = input("Enter Sentence1: ")
    #     sentence_p[i] = input("Enter Sentence2: ")
    #     sentence_o[i] = clean_text(sentence_o[i])
    #     sentence_p[i] = clean_text(sentence_p[i])

    # sentence_o[0] = "uk india business council chairperson said 60 british firms likely increase investment india next years firms positive reforms fdi introduced last two years added india even important britain said"
    # sentence_p[0] = "60 british firms to increase investment in india"
    # sentence_o[1] = "india 39 first cosmetics brand lakmé founded 1952 jrd tata subsidiary tata oil mills business tycoon born july 29 1904 requested prime minister manufacture beauty products country brand named french opera lakmé french form indian goddess"
    # sentence_p[1] = "jrd tata founded india 39 s first cosmetics brand"
    # sentence_o[0] = clean_text(sentence_o[0])
    # sentence_p[0] = clean_text(sentence_p[0])
    # sentence_o[1] = clean_text(sentence_o[1])
    # sentence_p[1] = clean_text(sentence_p[1])

    sen_p = []
    sen_p.append((clean_text(sen1), clean_text(sen2)))
    # for i in range(nnn):
    #     sen1 = sentence_o[i]
    #     sen2 = sentence_p[i]
    #     sen_p.append((sen1, sen2))

    id_pairs = get_pair_indices(vocab, sen_p)


    # prediction
    premise_seq    = [torch.tensor(seq[0]).long().to(device) for seq in id_pairs]
    hypothesis_seq = [torch.tensor(seq[1]).long().to(device) for seq in id_pairs]

    premise_len    = list(map(len, premise_seq))
    hypothesis_len = list(map(len, hypothesis_seq))

    batch = len(premise_seq)
    temp = pad_sequence(premise_seq + hypothesis_seq, batch_first=True)
    premise_seq    = temp[:batch, :]
    hypothesis_seq = temp[batch:, :]

    prediction = lstm_model([premise_seq, hypothesis_seq], premise_len, hypothesis_len)

    prediction_list = prediction.to('cpu')
    prediction_list = prediction_list.tolist()
    # print(len(prediction_list))

    # softmax_classes
    soft = torch.log_softmax(prediction, dim=1).argmax(dim=1)
    soft.tolist()
    labs = ""
    for i in soft:
        if i == 0:
            # labs.append('entailment')
            labs = labs + 'entailment'
        elif i == 1:
            # labs.append('neutral')
            labs = labs + 'neutral'
        else:
            # labs.append('contradiction')
            labs = labs + 'contradiction'
    # print(labs)

    return labs




class Summarizer(App):

    def build(self):

        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.8)
        self.window.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        self.lab1 = Label(
            text = "Enter Sentence to Summarize.....",
            font_size = 16,
            color = '#00FFCE'
        )
        self.window.add_widget(self.lab1)   
        self.sen = TextInput(
            multiline = False,
            padding_y = (10, 10),
            size_hint = (1, 1)
        )
        self.window.add_widget(self.sen) 

        self.button = Button(
            text = "Summarize",
            size_hint = (1, 0.5),
            bold = True,
            background_color = '#00FFCE'
        )

        self.lab3 = Label(
            text = "",
            font_size = 10,
            color = '#00FFCE'
        )
        self.window.add_widget(self.lab3)  

        self.button.bind(on_press = self.callback)
        self.window.add_widget(self.button)

        self.lab = Label(text = "")
        self.window.add_widget(self.lab)

        return self.window

    def callback(self, instance):
        # self.lab = Label(text = "\n>" + self.sen1.text + " >" + self.sen2.text)
        self.window.remove_widget(self.lab1)
        self.window.remove_widget(self.lab)
        self.window.remove_widget(self.sen)
        self.window.remove_widget(self.button)
        txt = str(summ(self.sen.text)[0]['summary_text'])
        txt_s = str(self.sen.text)

        self.premise = Label(text = txt_s)
        self.hypothesis = Label(text = txt)

        ctr = 0
        n = len(txt)
        for i in range(n):
            if txt[i] == " ":
                if ctr % 11 == 0:
                    ctr = 0
                    txt_str1 = txt[:i]
                    txt_str2 = txt[i+1:]
                    txt = str(txt_str1) + '\n' + str(txt_str2)
            ctr = ctr + 1   
        
        ctr = 0
        n = len(txt_s)
        for i in range(n):
            if txt_s[i] == " ":
                if ctr % 11 == 0:
                    ctr = 1
                    txt_str1 = txt_s[:i]
                    txt_str2 = txt_s[i+1:]
                    txt_s = str(txt_str1) + '\n' + str(txt_str2)
                ctr = ctr + 1
            ctr = ctr + 1   

        self.lab1 = Label(
            text = "Original Text.....",
            font_size = 18,
            color = '#00FFCE'
        )      
        self.window.add_widget(self.lab1)  
        self.lab = Label(text = txt_s)
        self.window.add_widget(self.lab)

        self.lab2 = Label(
            text = "Summarized Text (raw).....",
            font_size = 18,
            color = '#00FFCE'
        )      
        self.window.add_widget(self.lab2)  
        self.lab4 = Label(text = txt)
        self.window.add_widget(self.lab4)

        self.button_ent = Button(
            text = "Run Entailment",
            size_hint = (1, 0.5),
            bold = True,
            background_color = '#00FFCE'
        )

        self.lab_ = Label(
            text = "",
            font_size = 18,
            color = '#00FFCE' 
        )
        self.window.add_widget(self.lab_)
        self.label_ = Label(
            text = "",
            font_size = 14,
        )
        self.window.add_widget(self.label_)

        self.button_ent.bind(on_press = self.callback_ent)
        self.window.add_widget(self.button_ent)

    def callback_ent(self, instance):
        self.window.remove_widget(self.button_ent)

        self.window.remove_widget(self.label_)
        # self.lab_ = Label(text = ent(premise, hypothesis))
        self.label_ = Label(
            text = "Entailment label.....",
            font_size = 18,
            color = '#00FFCE' 
        )
        self.window.add_widget(self.label_)

        self.window.remove_widget(self.lab_)
        # self.lab_ = Label(text = ent(premise, hypothesis))
        self.lab_ = Label(
            text = ent(self.premise.text, self.hypothesis.text),
            font_size = 14,
        )
        self.window.add_widget(self.lab_)


if __name__ == "__main__":

    Summarizer().run()
