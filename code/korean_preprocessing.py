import numpy as np
import pandas as pd
import re
import json
import tensorflow as tf
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from utils.time_checking import timecheck

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print(e)

DATA_IN_PATH = './code/data/'
STOPWORDS_PATH = './code/data/korean_stopwords.txt'
DATA_CONFIGS = 'data_configs.json'

twitter = Okt()
train_data = pd.read_excel(DATA_IN_PATH + 'doc_set.xlsx')
stopwords = pd.read_csv(STOPWORDS_PATH)['stopwords'].tolist()
train_data = train_data.sample(frac=1)

def preprocessing(content, remove_stopwords=False):
    # remove_special_character
    tech_text = re.sub('[^가-힣A-z0-9]', ' ', content)
    tech_text = re.sub(r"\s{2,}", " ", tech_text)
    tech_text = re.sub("\n", " ", tech_text)

    if remove_stopwords:
        tech_words = tech_text.split()
        tech_words = [w for w in tech_words if (w not in stopwords)]
        clean_text = ' '.join(tech_words)
    else:
        clean_text = tech_text

    return tech_text

def remove_stopwords(text, nlp_tool, remove_stopwords=False, nouns=False):
    after_preprocess1 = preprocessing(text, remove_stopwords=remove_stopwords)
    if nouns:
        contents = nlp_tool.nouns(after_preprocess1)
    else:
        contents = nlp_tool.morphs(after_preprocess1, stem=True)
    stopwords = pd.read_csv(STOPWORDS_PATH)['stopwords'].tolist()
    clean_text = [token for token in contents if (token not in stopwords) and len(token) > 1]
    return clean_text

@timecheck
def clean_text_list(data):
    clean_train_contents = []
    for text in data['content']:
        if type(text) == str:
            clean_train_contents.append(remove_stopwords(text, nlp_tool=twitter, remove_stopwords=True, nouns=True))
        else:
            clean_train_contents.append([])
    return clean_train_contents


text_list = clean_text_list(train_data)

train_idx = round(len(text_list) * 0.8)
train_list = text_list[:train_idx]
test_list = text_list[train_idx:]

from sklearn.preprocessing import LabelEncoder
lbl_e = LabelEncoder()
text_label = lbl_e.fit_transform(train_data.new_class)
train_y_s = text_label[:train_idx]
test_y_s = text_label[train_idx:]

label = tf.keras.utils.to_categorical(text_label, num_classes = len(np.unique(train_data.new_class)))
train_y = label[:train_idx]
test_y = label[train_idx:]

len(train_y)
len(test_y)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_list)
train_sequence = tokenizer.texts_to_sequences(train_list)
test_sequence = tokenizer.texts_to_sequences(test_list)

sequence_data = dict()
sequence_data['train_seq'] = train_sequence
sequence_data['test_seq'] = test_sequence

word_idx = tokenizer.word_index

MAX_SEQUENCE_LENGTH = 174

train_input = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
train_labels = np.array(train_y)
test_input = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_labels = np.array(test_y)

TRAIN_INPUT_DATA = 'train_input.npy'
TRAIN_LABEL_DATA = 'train_label.npy'
TEST_INPUT_DATA = 'test_input.npy'
TEST_LABEL_DATA = 'test_label.npy'

TRAIN_LABEL_DATA_SPARSE = 'train_label_sparse.npy'
TEST_LABEL_DATA_SPARSE = 'test_label_sparse.npy'

DATA_CONFIGS = 'data_configs.json'
SEQ_CONFIGS = 'seq_configs.json'

data_configs = {}
data_configs['vocab'] = word_idx
data_configs['vocab_size'] = len(word_idx)


import os

DATA_IN_PATH = os.getcwd() + '\\tf_data\\'

# 전처리 된 데이터를 넘파이 형태로 저장
np.save(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'wb'), train_input)
np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'wb'), train_labels)
np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_input)
np.save(open(DATA_IN_PATH + TEST_LABEL_DATA, 'wb'), test_labels)

np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA_SPARSE, 'wb'), train_y_s)
np.save(open(DATA_IN_PATH + TEST_LABEL_DATA_SPARSE, 'wb'), test_y_s)
# 데이터 사전을 json 형태로 저장
json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'), ensure_ascii=False)
json.dump(sequence_data, open(DATA_IN_PATH + SEQ_CONFIGS, 'w'), ensure_ascii=False)
