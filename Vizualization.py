#!/usr/bin/env python
# coding: utf-8

# In[12]:


import requests
from bs4 import BeautifulSoup
from time import sleep
from itertools import groupby
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import pandas as pd

import nltk
from nltk import wordpunct_tokenize, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import pprint

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import csv

import time
import datetime
from datetime import datetime
from datetime import timezone

import sqlite3

from natasha import MorphVocab, Doc, NewsNERTagger, NewsMorphTagger, NewsEmbedding, Segmenter
from collections import Counter
import re, os
from stop_words import get_stop_words

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph_tagger = NewsMorphTagger(emb)

import plotly.graph_objects as go
import os
import orca
import plotly
import plotly.offline

import ast


# In[4]:


def retrieve_proc_text(media_id):
    
    '''Retrieve processed texts of news by media id 
    Param media_id: id of media (1 - tass; 2 - meduza; 3 - interfax; 4 - ria; 5 - mediazona; 6 - lenta.ru; 7 - rbc')
    '''
        
    sqlite_connection = sqlite3.connect('C:/Users/alfyn/ВКР/sqlite_python.db') #исправить на свой путь к базе
    cursor = sqlite_connection.cursor()
    
    print('База данных подключена к SQLite')
    
    cursor.execute('SELECT proc_text FROM temp_values where media_id =?', media_id)
    retrieved_texts = cursor.fetchall()
    
    return retrieved_texts

    sqlite_connection.close()    
    print('База данных закрыта')


# In[ ]:


def friq_graph(color, word, file_name):
    
    """The program calculates the frequency of entered by the user word in the text 
    and total number of words. 
    Then it calculates the ratio of word frequency to 1000 words and creates a graph and a table
    Param color: color of bars.
    """
    
    quans = [] # for frequancy of word from input 
    rel = [] # for ratio
    gen = [] # for total
    
    
    # all four loops are similar
    quan_tass = 0
    tass_words = 0
    for w in processed_words_tass:
        if w == str(word): 
            quan_tass = quan_tass + 1
        else:
            tass_words = tass_words + 1
    if quan_tass != 0:
        res1 = quan_tass / (tass_words + quan_tass) * 1000 # 'plus' to consider all words
        quans.append(res1)
        rel.append(quan_tass)
        gen.append(tass_words+quan_tass)
    else:
        quans.append(0)
        rel.append(0)
        gen.append(0)


    quan_meduza = 0
    meduza_words = 0
    for w in processed_words_meduza:
        if w == str(word):
            quan_meduza = quan_meduza + 1
        else:
            meduza_words = meduza_words + 1
    if quan_meduza != 0:
        res2 = quan_meduza / (meduza_words + quan_meduza) * 1000 # 'plus' to consider all words
        quans.append(res2)
        rel.append(quan_meduza)
        gen.append(meduza_words+quan_meduza)
    else:
        quans.append(0)
        rel.append(0)
        gen.append(0)
    
    quan_interfax = 0
    interfax_words = 0
    for w in processed_words_interfax:
        if w == str(word):
            quan_interfax = quan_interfax + 1
        else:
            interfax_words = interfax_words + 1
    if quan_interfax != 0:
        res3 = quan_interfax / (interfax_words + quan_interfax) * 1000 # 'plus' to consider all words
        quans.append(res3)
        rel.append(quan_interfax)
        gen.append(interfax_words+quan_interfax)
    else:
        quans.append(0)
        rel.append(0)
        gen.append(0)

    quan_ria = 0
    ria_words = 0
    for w in processed_words_ria:
        if w == str(word):
            quan_ria = quan_ria + 1
        else:
            ria_words = ria_words + 1
    if quan_ria != 0:
        res4 = quan_ria / (ria_words + quan_ria) * 1000 # 'plus' to consider all words
        quans.append(res4)
        rel.append(quan_ria)
        gen.append(ria_words+quan_ria)
    else:
        quans.append(0)
        rel.append(0)
        gen.append(0)

    quan_lenta = 0
    lenta_words = 0
    for w in processed_words_lenta:
        if w == str(word):
            quan_lenta = quan_lenta + 1
        else:
            lenta_words = lenta_words + 1
    if quan_lenta != 0:
        res5 = quan_lenta / (lenta_words + quan_lenta) * 1000 # 'plus' to consider all words
        quans.append(res5)
        rel.append(quan_lenta)
        gen.append(lenta_words+quan_lenta)
    else:
        quans.append(0)
        rel.append(0)
        gen.append(0)

    quan_rbc = 0
    rbc_words = 0
    for w in processed_words_rbc:
        if w == str(word):
            quan_rbc = quan_rbc + 1
        else:
            rbc_words = rbc_words + 1
    if quan_rbc != 0:
        res6 = quan_rbc / (rbc_words + quan_rbc) * 1000 # 'plus' to consider all words
        quans.append(res6)
        rel.append(quan_rbc)
        gen.append(rbc_words+quan_rbc)
    else:
        quans.append(0)
        rel.append(0)
        gen.append(0)

    if quans[0] != 0 and quans[1] != 0 and quans[2] != 0 and quans[3] != 0 and quans[4] != 0 and quans[5] != 0:
        lst = [['ТАСС', 'Медуза', 'Интарфакс', 'РИА новости', 'Лента.ru', 'РБК'], quans]
        lst2 = [['ТАСС', 'Медуза', 'Интарфакс', 'РИА новости', 'Лента.ru', 'РБК'], rel, gen]
        for_graph = pd.DataFrame(lst).transpose().sort_values(by = 1, ascending=False) # Data Frame for qraph
        for_graph.columns = ['СМИ', 'Частота употребления слова']


        plotly.offline.plot({"data": [go.Table(header=dict(values=['СМИ', 'Абсолютная частота', 'Количество слов']), cells=dict(values=lst2))],
                        "layout": go.Layout(title="Абсолютная частота слова")},
                        image='png', image_filename=file_name)

    
        sns.set(font = 'Verdana', font_scale=1.5, style = "white")

        fig_dims = (15, 9)
        fig, ax = plt.subplots(figsize=fig_dims, dpi=400, sharex=True)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.axes.xaxis.set_visible(False)

        graph = sns.barplot(x=for_graph['Частота употребления слова'], y=for_graph['СМИ'], color=color, ax=ax, orient='h')
        graph.set(xlabel=None, ylabel=None)
        for p in graph.patches:
            width = p.get_width()
            plt.text(0.1 + p.get_width(), p.get_y() + 0.55 * p.get_height(),
                    '{:1.2f}'.format(width).replace('.00', ''),
                    ha='center', va='center')

        fig.savefig("C:/Users/alfyn/ВКР/" + file_name)

        return graph

    else:
        error_message = 'К сожалению, сегодня такое слово не встречалось в новостях'

        return error_message


# In[17]:


def words_top(processed_words_media):
    words_list_start = processed_words_media
    words_list = []
    words_friq = {}

    for i in words_list_start:
        words_list.append(i.capitalize())

    for w in words_list:
        if w not in words_friq.keys():
            words_friq[w] = 1
        else:
            words_friq[w] = words_friq.get(w) + 1

    sorted_words_friq = sorted(words_friq.items(), key=lambda x: x[1], reverse=True)[0:15]

    return sorted_words_friq


# In[18]:


def graph_of_top(processed, color, file_name):
    df = pd.DataFrame(words_top(processed))
    df.columns = ['Слово', 'Количество повторений']

    sns.set(font='Verdana', font_scale=1.7, style="white")

    fig_dims = (17, 8)
    fig, ax = plt.subplots(figsize=fig_dims, dpi=400)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axes.xaxis.set_visible(False)

    graph = sns.barplot(x=df['Количество повторений'], y=df['Слово'], color=color, ax=ax, orient='h')
    graph.set(xlabel=None, ylabel=None)

    for p in graph.patches: #for annotate each bar in our plot
        width = p.get_width()
        plt.text(2 + p.get_width(), p.get_y() + 0.55 * p.get_height(),
                 '{:1.2f}'.format(width).replace('.00', ''),
                 ha='center', va='center')

    fig.savefig("C:/Users/alfyn/ВКР/" + file_name + '.png')

    return graph


# In[20]:


proc_text_all = ast.literal_eval(retrieve_proc_text('0')[0][0]) #сразу извлекаем список из строки
proc_text_tass = ast.literal_eval(retrieve_proc_text('1')[0][0])
proc_text_meduza = ast.literal_eval(retrieve_proc_text('2')[0][0])
proc_text_interfax = ast.literal_eval(retrieve_proc_text('3')[0][0])
proc_text_ria = ast.literal_eval(retrieve_proc_text('4')[0][0])
# proc_text_mediazona = ast.literal_eval(retrieve_proc_text('5')[0][0]) #исключаем пока медиазону, тк там мало слов
proc_text_lenta = ast.literal_eval(retrieve_proc_text('6')[0][0])
proc_text_rbc = ast.literal_eval(retrieve_proc_text('7')[0][0])


# In[ ]:


word = input('Введите слово c маленькой буквы: ')


# In[ ]:


friq_graph('lightskyblue', word, 'C:/Users/alfyn/ВКР/partuq_word_in_dif_media.png')


# In[23]:


words_top_tass = words_top(proc_text_tass)
words_top_meduza = words_top(proc_text_meduza)
words_top_interfax = words_top(proc_text_interfax)
words_top_ria = words_top(proc_text_ria)
words_top_lenta = words_top(proc_text_lenta)
words_top_rbc = words_top(proc_text_rbc)
words_top_all = words_top(proc_text_all)


# In[24]:


words_top_tass


# In[28]:


graph_of_top(proc_text_all, 'hotpink', 'C:/Users/alfyn/ВКР/top.png')

