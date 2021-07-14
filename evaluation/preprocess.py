import re
from string import punctuation
import json

def clean_text(text, stop_words = False, lemmatization = False):
  text = text.lower().split()
  if stop_words:
    stop = stopwords.words('english')
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
  
  text = " ".join(text)
  text = re.sub("[^A-Za-z']+", ' ', str(text)).replace("'", '')
  text = re.sub(r"\bum*\b", "", text)
  text = re.sub(r"\buh*\b", "", text)
  text = re.sub(r"won\'t", "will not", text)
  text = re.sub(r"can\'t", "can not", text)
  text = re.sub(r"can\'t", "can not", text)
  text = re.sub(r"n\'t", " not", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'s", " is", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'t", " not", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'m", " am", text)
  if lemmatization:
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])
  
  text = text.translate(str.maketrans('', '', punctuation))
  return text.strip()

# print(clean_text("Hello Worldly Beings"))

def preprocess_text(train_df, val_df):
    train_df.drop(['sentence1_binary_parse'], axis = 1, inplace=True)
    val_df.drop(['sentence1_binary_parse'], axis = 1, inplace=True)
    train_df.drop(['sentence2_binary_parse'], axis = 1, inplace=True)
    val_df.drop(['sentence2_binary_parse'], axis = 1, inplace=True)
    train_df.drop(['sentence1_parse'], axis = 1, inplace=True)
    val_df.drop(['sentence1_parse'], axis = 1, inplace=True)
    train_df.drop(['sentence2_parse'], axis = 1, inplace=True)
    val_df.drop(['sentence2_parse'], axis = 1, inplace=True)
    train_df.drop(['captionID'], axis = 1, inplace=True)
    val_df.drop(['captionID'], axis = 1, inplace=True)
    train_df.drop(['pairID'], axis = 1, inplace=True)
    val_df.drop(['pairID'], axis = 1, inplace=True)
    train_df.drop(['label1'], axis = 1, inplace=True)
    val_df.drop(['label1'], axis = 1, inplace=True)
    train_df.drop(['label2'], axis = 1, inplace=True)
    val_df.drop(['label2'], axis = 1, inplace=True)
    train_df.drop(['label3'], axis = 1, inplace=True)
    val_df.drop(['label3'], axis = 1, inplace=True)
    train_df.drop(['label4'], axis = 1, inplace=True)
    val_df.drop(['label4'], axis = 1, inplace=True)
    train_df.drop(['label5'], axis = 1, inplace=True)
    val_df.drop(['label5'], axis = 1, inplace=True)
    for i in range(len(train_df)):
        if train_df['gold_label'][i] == '-':
            # print(i)
            train_df['gold_label'][i] = None
            train_df['sentence1'][i] = None
            train_df['sentence2'][i] = None
    for i in range(len(val_df)):
        if val_df['gold_label'][i] == '-':
            # print(i)
            val_df['gold_label'][i] = None
            val_df['sentence1'][i] = None
            val_df['sentence2'][i] = None
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    # train_df = train_df[(train_df['sentence1'].str.split().str.len() > 0) & (train_df['sentence2'].str.split().str.len() > 0)]
    # val_df = val_df[(val_df['sentence1'].str.split().str.len() > 0) & (val_df['sentence2'].str.split().str.len() > 0)]
    return train_df, val_df

def read_jsonl_file(path):
  with open(path, 'r') as json_file:
    json_list = list(json_file)

  results = []
  for json_str in json_list:
      results.append(json.loads(json_str))
  return results

def get_word_map(count):  
  word_map = {}
  for num in count:
    if num in word_map:
      word_map[num] += 1
    else:
      word_map[num] = 1

  return word_map