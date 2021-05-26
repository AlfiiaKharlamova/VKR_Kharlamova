#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
from time import sleep
from itertools import groupby
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
from rake_nltk import Metric, Rake
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from natasha import MorphVocab, Doc, NewsNERTagger, NewsMorphTagger, NewsEmbedding, Segmenter
from collections import Counter
import re, os
from stop_words import get_stop_words

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph_tagger = NewsMorphTagger(emb)
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
ner_tagger1 = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)


# In[63]:


import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go

import time
import datetime
from datetime import datetime
from datetime import timezone
import sqlite3

import nltk
from nltk import wordpunct_tokenize, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer


# In[7]:


def retrieve_by_date_and_media(condition_list, db_path):
    
    '''Retrieve text of news of current date
    Param unixdate: Unix datetime of today (list of first and last second of today)
    Param media: media_id that we want to retrieve 
    (1 - tass; 2 - meduza; 3 - interfax; 4 - ria; 5 - mediazona; 6 - lenta.ru; 7 - rbc')
    Param db_path: Path to database
    '''
        
    sqlite_connection = sqlite3.connect(db_path)
    cursor = sqlite_connection.cursor()
    
    print('База данных подключена к SQLite')
    
    cursor.execute('SELECT text FROM media_news where unixdate>? and unixdate<? and media_id =?', condition_list)
    retrieved_texts = cursor.fetchall()
    
    return retrieved_texts

    sqlite_connection.close()    
    print('База данных закрыта')


# In[8]:


def getting_condition_list(media):
    
    '''Generating todays' unix datetime (list of the first and the last second of today + media_id)
    Param media: media_id for retrieving data by unix datetime and media_id
    '''
    
    current_unix = []
    
    first_date = datetime.today().strftime('%Y-%m-%d')+'-00:00:00'
    first_unixtime = time.mktime(datetime.strptime(first_date, "%Y-%m-%d-%H:%M:%S").timetuple())
    current_unix.append(first_unixtime)
    
    second_date = datetime.today().strftime('%Y-%m-%d')+'-23:59:59'
    second_unixtime = time.mktime(datetime.strptime(second_date, "%Y-%m-%d-%H:%M:%S").timetuple())
    current_unix.append(second_unixtime)
    
    current_unix.append(media)
    
    return (current_unix)


# In[9]:


condition_list_tass = getting_condition_list(1)
condition_list_meduza = getting_condition_list(2)
condition_list_interfax = getting_condition_list(3)
condition_list_ria = getting_condition_list(4)
condition_list_mediazona = getting_condition_list(5)
condition_list_lenta = getting_condition_list(6)
condition_list_rbc = getting_condition_list(7)


# In[12]:


db_col_meduza = retrieve_by_date_and_media(condition_list_meduza, 'C:/Users/alfyn/ВКР/sqlite_python.db') #заменить на свой путь к базе
db_col_tass = retrieve_by_date_and_media(condition_list_tass, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_interfax = retrieve_by_date_and_media(condition_list_interfax, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_ria = retrieve_by_date_and_media(condition_list_ria, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_mediazona = retrieve_by_date_and_media(condition_list_mediazona, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_lenta = retrieve_by_date_and_media(condition_list_lenta, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_rbc = retrieve_by_date_and_media(condition_list_rbc, 'C:/Users/alfyn/ВКР/sqlite_python.db')


# In[25]:


final_text_all = ''

for i in db_col_meduza:
    final_text_all = i[0] + final_text_all
for i in db_col_tass:
    final_text_all = i[0] + final_text_all
for i in db_col_interfax:
    final_text_all = i[0] + final_text_all
for i in db_col_mediazona:
    final_text_all = i[0] + final_text_all
for i in db_col_ria:
    final_text_all = i[0] + final_text_all
for i in db_col_lenta:
    final_text_all = i[0] + final_text_all
for i in db_col_rbc:
    final_text_all = i[0] + final_text_all


# In[27]:


doc = Doc(final_text_all)


# In[28]:


doc.segment(segmenter)
display(doc.sents[:5])


# In[29]:


new_sent = doc.sents
new_sent_list = [_.text for _ in new_sent]


# In[69]:


#create new list of NER in each sentence
list_of_ner = []
for s in new_sent_list:
    list = []
    doc1 = Doc(s)
    doc1.segment(segmenter)
    doc1.tag_ner(ner_tagger)
    doc1.tag_morph(morph_tagger)
    for span in doc1.spans:
        span.normalize(morph_vocab)
        if(span.type == 'PER'):
            list.append(span.normal)
    list_of_ner.append(list)


# In[135]:


important_per = {'Путин': 'Владимир Путин', 'Байден': 'Джо Байден', 'Трамп': 'Дональд Трамп', 
                 'Навальный': 'Алексей Навальный', 'Маск': 'Илон Маск', 'Протасевич': 'Роман Протасевич'}


# In[344]:


#удаляем просто имена и просто фамилии
new_list_of_ner = []

for i in list_of_ner:
    new_i = [] #заводим чистый список под каждый новый элемент
    for f in i: #начинаем итерацию 
        
        if f in important_per.keys(): #заходим в f, если там  ТОЛЬКО фамилия важной персоны, то заменем ее на Имя Фамилия
            new_i.append(important_per.get(f)) 

        f = nltk.word_tokenize(f) # продолжаем ту же итерацию - делим каждый элемент старого списка на слова.
        if len(f) >= 2: 
            new_i.append(' '.join(f)) #если слов >=2, то добавляем их в новый список, если меньше, то ничего не делаем
    new_list_of_ner.append(new_i) #в конце в общий новый список добавляем список сущностей в рамках одного предложения
    
for i in new_list_of_ner:
    if len(i) == 0:
        new_list_of_ner.remove(i)

#вручную переводим некоторые именованные сущности в нужную словарную форму
for i in new_list_of_ner:
    for f in i:
        if f == 'София Сапеги':
            i.remove(f)
            i.append('София Сапега')
        if f == 'Илона Маска':
            i.remove(f)
            i.append('Илон Маск')
        if f == 'Джозеф Байден': 
            i.remove(f)
            i.append('Джо Байден')
        if f == 'Алексей Навального': 
            i.remove(f)
            i.append('Алексей Навальный')


# In[345]:


new_list_of_ner


# In[346]:


def preprocess_words(corpus):
    doc = Doc(corpus)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    lemmas = []
    stop_words = get_stop_words('russian')

    for token in doc.tokens:
        if token.lemma not in stop_words and not re.match('\W+', token.lemma):
            lemmas.append(token.lemma)
    return lemmas

def ner_graph(ner_sents):
    pairs = []

    for sent in ner_sents:
        if len(sent) > 1:
            for i in range(len(sent)-1):
                for j in range(i, len(sent)):
                    if i != j:
                        pair = min(sent[i], sent[j]) + '_' + max(sent[i], sent[j])
                        pairs.append(pair)
    
    counts = Counter(pairs).most_common()
    return counts

def write_result(counts, filename):
    #записываем результат в csv
    csv = 'word1,word2,weight' + '\n'
    
    for count in counts:
        word1 = count[0].split('_')[0]
        word2 = count[0].split('_')[1]
        weight = str(count[1])
        csv += word1 + ';' + word2 + ';' + weight + '\n'
        
        
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(csv)
        
def make_word_graph(lemmas):
    pairs = []

    for i in range(len(lemmas)-1):
        pair = min(lemmas[i], lemmas[i+1]) + '_' + max(lemmas[i], lemmas[i+1])
        pairs.append(pair)

    counts = Counter(pairs).most_common()
    return counts


# In[147]:


ner_counts = ner_graph(new_list_of_ner) #ner_counts подаем на вход функции write_result
ner_counts_new = ner_counts[35:]
print(ner_counts)


# In[140]:


text_lemmas = preprocess_words(final_text_all)


# In[141]:


write_result(ner_counts, 'ner_graph.csv')


# #### Рисуем граф

# In[405]:


Data = open('ner_graph.csv', "r", encoding='utf8')
next(Data, None)  # skip the first line in the input file
Graphtype = nx.Graph()

G = nx.parse_edgelist(Data, delimiter=';', create_using=Graphtype,
                      nodetype=str, data=(('weight', float),))


# In[406]:


labels = {}
for node in G.nodes():
    labels[node] = node
nodes_list = G.nodes()
edges_list = G.edges(data=True)
print(nodes_list)


# In[214]:


edge_weight = []
for edge in G.edges():
    edge_weight.append(G.edges()[edge]['weight'])


# In[461]:


coord = nx.spring_layout(G, k = 0.4, iterations = 4)


# In[462]:


fig = plt.figure(figsize=(40,25))

labs = nx.draw_networkx_labels(G, coord, labels, font_size = 20, font_color = '#000000', font_family= 'Verdana', 
                               font_weight = 'black', alpha = 0.5)
nodes = nx.draw_networkx_nodes(G, label = labels, nodelist = nodes_list, pos = coord, node_size = 4000, node_color = "#B5DDA4")
edges = nx.draw_networkx_edges(G, edgelist = edges_list, pos = coord, width = edge_weight, edge_color = "#B5DDA4")
limits = plt.axis("off")
fig.savefig("C:/Users/alfyn/ВКР/" + 'network' + '.png')


# In[410]:


from networkx.algorithms import community
communities_generator = community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
sorted(map(sorted, next_level_communities))

