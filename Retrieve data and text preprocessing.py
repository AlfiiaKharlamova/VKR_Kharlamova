#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
from bs4 import BeautifulSoup
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

import nltk
from nltk import wordpunct_tokenize, word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import pprint

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


# In[3]:


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


# In[4]:


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


# In[5]:


stopwords = ['а', 'в', 'г', 'е', 'ж', 'и', 'к', 'м', 'о', 'об', 'с', 'т', 'у', 'я', 'бы', 'во', 'вы', 'да', 'до', 
             'ее', 'ей', 'ею', 'её', 'же', 'за', 'из', 'им', 'их', 'ли', 'мы', 'на', 'не', 'ни', 'но', 'ну', 
             'них', 'об', 'он', 'от', 'по', 'со', 'та', 'те', 'то', 'ту', 'ты', 'уж', 'без', 'был', 'вам', 'вас', 
             'ваш', 'вон', 'вот', 'все', 'всю', 'вся', 'всё', 'где', 'год', 'два', 'две', 'дел', 'для', 'его', 'ему', 
             'еще', 'ещё', 'или', 'ими', 'имя', 'как', 'кем', 'ком', 'кто', 'лет', 'мне', 'мог', 'мож', 'мои', 'мой', 
             'мор', 'моя', 'моё', 'над', 'нам', 'нас', 'наш', 'нее', 'ней', 'нем', 'нет', 'нею', 'неё', 'них', 'оба', 
             'она', 'они', 'оно', 'под', 'пор', 'при', 'про', 'раз', 'сам', 'сих', 'так', 'там', 'тем', 'тех', 'том', 
             'тот', 'тою', 'три', 'тут', 'уже', 'чем', 'что', 'эта', 'эти', 'это', 'эту', 'алло', 'буду', 'будь', 'бывь', 
             'была', 'были', 'было', 'быть', 'вами', 'ваша', 'ваше', 'ваши', 'ведь', 'весь', 'вниз', 'всем', 'всех', 
             'всею', 'года', 'году', 'даже', 'двух', 'день', 'если', 'есть', 'зато', 'кого', 'кому', 'куда', 'лишь', 
             'люди', 'мало', 'меля', 'меня', 'мимо', 'мира', 'мной', 'мною', 'мочь', 'надо', 'нами', 'наша', 'наше', 
             'наши', 'него', 'нему', 'ниже', 'ними', 'один', 'пока', 'пора', 'пять', 'рано', 'сама', 'сами', 'само', 
             'саму', 'свое', 'свои', 'свою', 'себе', 'себя', 'семь', 'стал', 'суть', 'твой', 'твоя', 'твоё', 'тебе', 
             'тебя', 'теми', 'того', 'тоже', 'тому', 'туда', 'хоть', 'хотя', 'чаще', 'чего', 'чему', 'чтоб', 'чуть', 
             'этим', 'этих', 'этой', 'этом', 'этот', 'более', 'будем', 'будет', 'будто', 'будут', 'вверх', 'вдали', 'вдруг', 
             'везде', 'внизу', 'время', 'всего', 'всеми', 'всему', 'всюду', 'давно', 'даром', 'долго', 'друго', 'жизнь', 
             'занят', 'затем', 'зачем', 'здесь', 'иметь', 'какая', 'какой', 'когда', 'кроме', 'лучше', 'между', 'менее', 
             'много', 'могут', 'может', 'можно', 'можхо', 'назад', 'низко', 'нужно', 'одной', 'около', 'опять', 'очень', 
             'перед', 'позже', 'после', 'потом', 'почти', 'пятый', 'разве', 'рядом', 'самим', 'самих', 'самой', 'самом', 
             'своей', 'своих', 'свой', 'снова', 'собой', 'собою', 'такая', 'также', 'такие', 'такое', 'такой', 'тобой', 
             'тобою', 'тогда', 'тысяч', 'уметь', 'часто', 'через', 'чтобы', 'шесть', 'этими', 'этого', 'этому', 'близко', 
             'больше', 'будете', 'будешь', 'бывает', 'важная', 'важное', 'важные', 'важный', 'вокруг', 'восемь', 'всегда', 
             'второй', 'далеко', 'дальше', 'девять', 'десять', 'должно', 'другая', 'другие', 'других', 'другое', 'другой', 
             'занята', 'занято', 'заняты', 'значит', 'именно', 'иногда', 'каждая', 'каждое', 'каждые', 'каждый', 'кругом', 
             'меньше', 'начала', 'нельзя', 'нибудь', 'никуда', 'ничего', 'обычно', 'однако', 'одного', 'отсюда', 'первый', 
             'потому', 'почему', 'просто', 'против', 'раньше', 'самими', 'самого', 'самому', 'своего', 'сейчас', 'сказал', 
             'совсем', 'теперь', 'только', 'третий', 'хорошо', 'хотеть', 'хочешь', 'четыре', 'шестой', 'восьмой', 'впрочем', 
             'времени', 'говорил', 'говорит', 'девятый', 'десятый', 'кажется', 'конечно', 'которая', 'которой', 'которые', 
             'который', 'которых', 'наверху', 'наконец', 'недавно', 'немного', 'нередко', 'никогда', 'однажды', 'посреди', 
             'сегодня', 'седьмой', 'сказала', 'сказать', 'сколько', 'слишком', 'сначала', 'спасибо', 'человек', 'двадцать', 
             'довольно', 'которого', 'наиболее', 'недалеко', 'особенно', 'отовсюду', 'двадцатый', 'миллионов', 'несколько', 
             'прекрасно', 'процентов', 'четвертый', 'двенадцать', 'непрерывно', 'пожалуйста', 'пятнадцать', 'семнадцать', 
             'тринадцать', 'двенадцатый', 'одиннадцать', 'пятнадцатый', 'семнадцатый', 'тринадцатый', 'шестнадцать', 
             'восемнадцать', 'девятнадцать', 'одиннадцатый', 'четырнадцать', 'шестнадцатый', 'восемнадцатый', 'девятнадцатый', 
             'действительно', 'четырнадцатый', 'многочисленная', 'многочисленное', 'многочисленные', 'многочисленный', 
             'январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 
             'декабрь', 'янв', 'фев', 'мар', 'апр', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек', 'москва', 'тасс', 'число',
             'область', 'сутки', 'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье', 'россия', 
             'московский', 'столица', 'регион', 'страна', 'деятельность', 'тысяча', 'рф', 'программа', 'заявить', 'пояснить', 
             'прокомментировать', 'смочь', 'российский', 'ситуация', 'гражданин', 'неделя', 'месяц', 'большой', 'федерация', 
             'ыидеть', 'коллега', 'слово', 'город', 'правило', 'хороший', 'многие', 'стать', 'организация', 'движение', 
             'компания', 'минута', 'час', 'год', 'сообщить', 'сообщать', 'возрасти', 'подтвердить', 'новый', 'дать', 'начать', 
             'ранее', 'риа', 'новость', 'медуза', 'lenta', 'лента', 'рбк', 'дело', 'случай', 'глава', 'должный', 'отметить', 
             'ноходится', 'находиться', 'объявить', 'сотрудник', 'рубль', 'получить', 'interfax.ru', 'interfax', 'интерфакс', 
             'сообщение', 'миллиард', 'решение', 'суд', 'место', 'президент', 'власть', 'ребенок', 'люди', 'человек' 'ребёнок', 
             'сообщаться', 'проект', 'группа', 'считать', 'результат', 'участие', 'говориться', 'говорится', 'данные', 
             'начало', 'мера', 'территория', 'состояние', 'россиянин', 'работа', 'рассказать', 'новое', 'уровень', 
             'правительство', 'управление', 'отношение', 'чехия', 'представитель']

punct = ['.', ',', '—' ,'«', '»', '"', '!', '?', ':', ';', '(', ')', '-', '&', '``', '<', '>', "''", '/', '[', ']', '|', '$',
         '#', '{', '}', '@', '%', '*', '№']

punct_for_graph = ['.', ',','«', '»', '"', '!', '?', ':', ';', '(', ')', '&', '``', '<', '>', "''", '/', '[', ']', '|', '$',
         '#', '{', '}', '@', '%', '*']


# In[6]:


def preprocess_text(db_col):
    
    '''Preprocessing text data for future use: tokenizing, removing punctuation and stop words, removing words form 
    Param db_col: list of texts that we got from the database
    '''
    
    text = ''
    for i in db_col:
        text = text + i[0]
    tokens = nltk.word_tokenize(text)
    
    list_of_words = []

    for l in punct:
        for i in tokens:
            if l in i:
                tokens.remove(i)
    
    for word in tokens:
        word.lower()
        p = morph.parse(word)[0] 
        list_of_words.append(p.normal_form)

    for l in stopwords:
        for i in list_of_words:
            if i in l:
                list_of_words.remove(i)
    
    list_of_words
    final_list = [x for x in list_of_words if not (x.isdigit() 
                                         or x[0] == '-' and x[1:].isdigit())]
    
    return final_list


# In[8]:


condition_list_tass = getting_condition_list(1)
condition_list_meduza = getting_condition_list(2)
condition_list_interfax = getting_condition_list(3)
condition_list_ria = getting_condition_list(4)
condition_list_mediazona = getting_condition_list(5)
condition_list_lenta = getting_condition_list(6)
condition_list_rbc = getting_condition_list(7)


# In[9]:


db_col_meduza = retrieve_by_date_and_media(condition_list_meduza, 'C:/Users/alfyn/ВКР/sqlite_python.db') #заменить на свой путь к базе
db_col_tass = retrieve_by_date_and_media(condition_list_tass, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_interfax = retrieve_by_date_and_media(condition_list_interfax, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_ria = retrieve_by_date_and_media(condition_list_ria, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_mediazona = retrieve_by_date_and_media(condition_list_mediazona, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_lenta = retrieve_by_date_and_media(condition_list_lenta, 'C:/Users/alfyn/ВКР/sqlite_python.db')
db_col_rbc = retrieve_by_date_and_media(condition_list_rbc, 'C:/Users/alfyn/ВКР/sqlite_python.db')


# In[10]:


processed_words_tass = preprocess_text(db_col_tass)
processed_words_meduza = preprocess_text(db_col_meduza)
processed_words_interfax = preprocess_text(db_col_interfax)
processed_words_ria = preprocess_text(db_col_ria)
processed_words_mediazona = preprocess_text(db_col_mediazona)
processed_words_lenta = preprocess_text(db_col_lenta)
processed_words_rbc = preprocess_text(db_col_rbc)


# In[12]:


processed_words_all = processed_words_tass

for w in processed_words_meduza:
    processed_words_all.append(w)

for w in processed_words_interfax:
    processed_words_all.append(w)

for w in processed_words_ria:
    processed_words_all.append(w)
    
for w in processed_words_lenta:
    processed_words_all.append(w)
    
for w in processed_words_rbc:
    processed_words_all.append(w)


# In[20]:


processed_tass_db = [1, str(processed_words_tass)]
processed_meduza_db = [2, str(processed_words_meduza)]
processed_interfax_db = [3, str(processed_words_interfax)]
processed_ria_db = [4, str(processed_words_ria)]
processed_mediazona_db = [5, str(processed_words_mediazona)]
processed_lenta_db = [6, str(processed_words_lenta)]
processed_rbc_db = [7, str(processed_words_rbc)]
processed_rbc_all = [0, str(processed_words_rbc)]


# In[27]:


def saving_data_to_db(proc_data):
    cursor.execute('INSERT INTO temp_values (media_id, proc_text) VALUES (?,?)', proc_data)
    print('Insert done')


# In[28]:


sqlite_connection = sqlite3.connect('C:/Users/alfyn/ВКР/sqlite_python.db') #заменить на свой путь к базе
print('База данных подключена к SQLite')
cursor = sqlite_connection.cursor()

cursor.execute('DELETE FROM temp_values')

saving_data_to_db(processed_tass_db)
saving_data_to_db(processed_meduza_db)
saving_data_to_db(processed_interfax_db)
saving_data_to_db(processed_ria_db)
saving_data_to_db(processed_mediazona_db)
saving_data_to_db(processed_lenta_db)
saving_data_to_db(processed_rbc_db)
saving_data_to_db(processed_rbc_all)

sqlite_connection.commit()
print('Изменения сохранены')
    
sqlite_connection.close()    
print('База данных закрыта')

