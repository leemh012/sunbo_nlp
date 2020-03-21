import numpy as np
import pandas as pd
import re
import json
import tensorflow as tf
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

DATA_IN_PATH = './data/'
STOPWORDS_PATH = './data/korean_stopwords.txt'
DATA_CONFIGS = 'data_configs.json'

twitter = Okt()
train_data = pd.read_excel(DATA_IN_PATH + '/doc_set.xlsx')
train_data.drop('Unnamed: 0', axis=1, inplace=True)
stopwords_path = './data/korean_stopwords.txt'
stopwords = pd.read_csv(stopwords_path)['stopwords'].tolist()


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

def remove_stopwords(text, nlp_tool, remove_stopwords=False):
    after_preprocess1 = preprocessing(text, remove_stopwords=remove_stopwords)
    contents = nlp_tool.morphs(after_preprocess1, stem=True)
    stopwords = pd.read_csv(STOPWORDS_PATH)['stopwords'].tolist()
    clean_text = [token for token in contents if (token not in stopwords) and len(token) > 1]
    return clean_text

clean_token = remove_stopwords(train_data['content'][0], nlp_tool=twitter, remove_stopwords=True)

clean_train_contents = []
for text in train_data['content']:
    if type(text) == str:
        clean_train_contents.append(remove_stopwords(text, nlp_tool=twitter, remove_stopwords=True))
    else:
        clean_train_contents.append([])


cleant_train_contents[:4]

len(cleant_train_contents)
len(train_data.new_class)


tokenizer = Tokenizer()
