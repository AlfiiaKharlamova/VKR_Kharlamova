#!/usr/bin/env python
# coding: utf-8

# In[9]:


import telebot
from telebot import types

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

import time
import datetime
from datetime import datetime
from datetime import timezone

import sqlite3

from natasha import MorphVocab, Doc, NewsNERTagger, NewsMorphTagger, NewsEmbedding, Segmenter
from collections import Counter

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph_tagger = NewsMorphTagger(emb)

import plotly.graph_objects as go
import os
import plotly
import plotly.offline

import string
import random


# In[3]:


bot = telebot.TeleBot('1760543777:AAEUPuGgD9wqIqSjlBtrKLWCtfRr3qz6LWE')


# In[4]:


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Привет, я бот Федор! "
                                               "Я генерирую частотные графики, основанные на словах, "
                                               "которые употребляются в новостях крупных российских интернет-СМИ.\n"
                                               "\n"
                                               "Чтобы узнать подробнее введите /help")
        
        keyboard = types.InlineKeyboardMarkup() #наша клавиатура
        key_plot_1 = types.InlineKeyboardButton(text='Ввести слово и увидеть частоту его использования', callback_data='graph_1') #кнопка «Да»
        keyboard.add(key_plot_1) #добавляем кнопку в клавиатуру
        key_plot_2 = types.InlineKeyboardButton(text='Выбрать СМИ и увидеть ТОП употребленных слов', callback_data='graph_2')
        keyboard.add(key_plot_2)
        key_plot_3 = types.InlineKeyboardButton(text='ТОП употребленных слов по всем СМИ', callback_data='graph_3')
        keyboard.add(key_plot_3)
        bot.send_message(message.chat.id, "Какой график вы хотите увидеть?", reply_markup=keyboard)
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Данный Telegram-бот служит для генерации частотных графиков, "
                                               "основанных на словах, которые употребляются в новостях крупных "
                                               "российских интернет-СМИ. \n"
                                               "Здесь вы можете увидеть:\n"
                                               "1. График частоты использования любого интересующего вас слова в шести СМИ "
                                               "(ТАСС, Медуза, Лента.ru, Интерфакс, РИА новости, РБК) \n"
                                               "2. График ТОП-15 слов в одном из выбранных вами СМИ \n"
                                               "3. График ТОП-15 слов по всем указанным СМИ \n"
                                               "Визуализация осуществляется на основе данных, собранных алгоритмом "
                                               "с главных страниц указанных СМИ\n"
                                               "\n"
                                               "Чтобы запустить бота, необходимо ввести текст '/start' в поле "
                                               "для ввода сообщений")
    elif message.text == "/feedback":
        f = bot.send_message(message.from_user.id, "Спасибо, что решили оставить отзыв! С вашей помощью я становлюсь лучше. Напишите отзыв:")
        bot.register_next_step_handler(f, saving_feedback)


def retrieve_by_date_and_media(condition_list):
    
    '''Retrieve text of news of current date
    Param unixdate: Unix datetime of today (list of first and last second of today)
    Param media: media_id that we want to retrieve 
    (1 - tass; 2 - meduza; 3 - interfax; 4 - ria; 5 - mediazona; 6 - lenta.ru; 7 - rbc')
    '''
        
    sqlite_connection = sqlite3.connect('C:/Users/alfyn/ВКР/sqlite_python.db')
    cursor = sqlite_connection.cursor()
    
    print('База данных подключена к SQLite')
    
    cursor.execute('SELECT text FROM media_news where unixdate>? and unixdate<? and media_id =?', condition_list)
    retrieved_texts = cursor.fetchall()
    
    return retrieved_texts


def getting_condition_list_user(media, date):
    '''Generating todays' unix datetime (list of the first and the last second of today + media_id)
    Param media: media_id for retrieving data by unix datetime and media_id
    Param date: date in 'YYYY-MM-DD' format
    '''

    current_unix = []
    user_date = str(date)

    first_date = user_date + '-00:00:00'
    first_unixtime = time.mktime(datetime.strptime(first_date, "%Y-%m-%d-%H:%M:%S").timetuple())
    current_unix.append(first_unixtime)

    second_date = user_date + '-23:59:59'
    second_unixtime = time.mktime(datetime.strptime(second_date, "%Y-%m-%d-%H:%M:%S").timetuple())
    current_unix.append(second_unixtime)

    current_unix.append(media)

    return (current_unix)


us_date = datetime.today().strftime('%Y-%m-%d')

condition_list_tass = getting_condition_list_user(1, us_date)
condition_list_meduza = getting_condition_list_user(2, us_date)
condition_list_interfax = getting_condition_list_user(3, us_date)
condition_list_ria = getting_condition_list_user(4, us_date)
condition_list_mediazona = getting_condition_list_user(5, us_date)
condition_list_lenta = getting_condition_list_user(6, us_date)
condition_list_rbc = getting_condition_list_user(7, us_date)

db_col_meduza = retrieve_by_date_and_media(condition_list_meduza)
db_col_tass = retrieve_by_date_and_media(condition_list_tass)
db_col_interfax = retrieve_by_date_and_media(condition_list_interfax)
db_col_ria = retrieve_by_date_and_media(condition_list_ria)
db_col_mediazona = retrieve_by_date_and_media(condition_list_mediazona)
db_col_lenta = retrieve_by_date_and_media(condition_list_lenta)
db_col_rbc = retrieve_by_date_and_media(condition_list_rbc)

#
#def generate_data(date):
#    global condition_list_tass
#    global condition_list_meduza
#    global condition_list_interfax
#    global condition_list_ria
#    global condition_list_mediazona
#    global condition_list_lenta
#    global condition_list_rbc
#    global db_col_meduza
#    global db_col_tass
#    global db_col_interfax
#    global db_col_ria
#    global db_col_mediazona
#    global db_col_lenta
#   global db_col_rbc

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
             'сообщение', 'миллиард', 'решение', 'суд', 'место', 'президент', 'власть', 'люди', 'человек',
             'сообщаться', 'проект', 'группа', 'считать', 'результат', 'участие', 'говориться', 'говорится', 'данные',
             'начало', 'мера', 'территория', 'состояние', 'россиянин', 'работа', 'рассказать', 'новое', 'уровень',
             'правительство', 'управление', 'отношение', 'чехия', 'представитель', 'владимир', 'алексей', 'сша', 'млрд',
             'млн', 'миллиард', 'миллион', 'тысячи', 'тысяча']

punct = ['.', ',', '—' ,'«', '»', '"', '!', '?', ':', ';', '(', ')', '-', '&', '``', '<', '>', "''", '/', '[', ']', '|', '$',
         '#', '{', '}', '@', '%', '*']

punct_for_graph = ['.', ',','«', '»', '"', '!', '?', ':', ';', '(', ')', '&', '``', '<', '>', "''", '/', '[', ']', '|', '$',
         '#', '{', '}', '@', '%', '*']


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

    final_list = [x for x in list_of_words if not (x.isdigit() 
                                         or x[0] == '-' and x[1:].isdigit())]
    
    return final_list


processed_words_tass = preprocess_text(db_col_tass)
# переделать функциюм generate_data чтобы она принимала два аргумента: id СМИ(или любой ключ уникальный для сми) и дату
# и перегенеривала данные отдельно по конкретному сми
# если processed_words_tass пустой, перегенериваем db_col_tass через вызов функции generate_data от вчерашнего дня
# и так для каждого сми
processed_words_meduza = preprocess_text(db_col_meduza)
processed_words_interfax = preprocess_text(db_col_interfax)
processed_words_ria = preprocess_text(db_col_ria)
processed_words_mediazona = preprocess_text(db_col_mediazona)
processed_words_lenta = preprocess_text(db_col_lenta)
processed_words_rbc = preprocess_text(db_col_rbc)

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


def freq_graph(color, word, file_name):
    
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

    # fig_2 = go.Figure(data=[go.Table(header=dict(values=['СМИ', 'Абсолютная частота', 'Количество слов']),
    #                                cells=dict(values=lst2))])

        plotly.offline.plot({"data": [go.Table(header=dict(values=['СМИ', 'Абсолютная частота', 'Количество слов']), cells=dict(values=lst2))],
                        "layout": go.Layout(title="Абсолютная частота слова")},
                        image='png', image_filename=file_name)

    # fig_2.write_image("C:/Users/alfyn/ВКР/partuq_word_in_dif_media_table.png")

    
        sns.set(font = 'Verdana', font_scale=1.5, style = "white")

        fig_dims = (15, 9)
        fig, ax = plt.subplots(figsize=fig_dims, dpi=400, sharex=True)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        graph = sns.barplot(x=for_graph['Частота употребления слова'], y=for_graph['СМИ'], color=color, ax=ax, orient='h')
        graph.set(xlabel=None, ylabel=None)
        #for p in graph.patches:
        #    width = p.get_width()
        #    plt.text(0.1 + p.get_width(), p.get_y() + 0.55 * p.get_height(),
        #            '{:1.2f}'.format(width).replace('.00', ''),
        #            ha='center', va='center')

        plt.title(f'Частота употребления слова {word.capitalize()} на 1000 слов', fontdict={'fontweight': 'bold'}, loc='left', pad=30)
        plt.annotate('*Обращаем ваще внимание, что с 23 апреля 2021 года Медуза признана иностанным агентом \n'
                     'source: telegram-канал "Слово дня" (t.me/novosti_slovo_dnya)', (0,0), (-10, -80), fontsize=16,
                     xycoords='axes fraction', textcoords='offset points', va='bottom')

        fig.savefig("C:/Users/alfyn/ВКР/" + file_name)

        return graph

    else:
        error_message = 'К сожалению, сегодня такое слово не встречалось в новостях'

        return error_message


def words_top(processed_words_media):
    words_list_start = processed_words_media
    words_list = []
    words_freq = {}

    for i in words_list_start:
        words_list.append(i.capitalize())

    for w in words_list:
        if w not in words_freq.keys():
            words_freq[w] = 1
        else:
            words_freq[w] = words_freq.get(w) + 1

    sorted_words_freq = sorted(words_freq.items(), key=lambda x: x[1], reverse=True)[0:15]

    return sorted_words_freq

def graph_of_top(processed, color, file_name, media_name):
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
        bar_an ='{:.0f}'.format(p.get_width())
        width, height = p.get_width(), p.get_height()
        x = p.get_x()+width+0.2
        y = p.get_y()+height/1.5
        ax.annotate(bar_an, (x, y), fontsize=14)

    if media_name == 'все':
        plt.title('ТОП-15 слов в ТАСС, Медузе, Интерфаксе, РИА новостях, РБК, Ленте.ru', fontdict={'fontweight': 'bold'},  loc='left', pad=30)
    elif media_name == 'тасс':
        plt.title('ТОП-15 слов в ТАСС', fontdict={'fontweight': 'bold'}, loc='left', pad=30)
    elif media_name == 'медуза':
        plt.title('ТОП-15 слов в Медузе (является иностранным агентом)', fontdict={'fontweight': 'bold'}, loc='left', pad=30)
    elif media_name == 'интерфакс':
        plt.title('ТОП-15 слов в Интерфакс', fontdict={'fontweight': 'bold'}, loc='left', pad=30)
    elif media_name == 'рбк':
        plt.title('ТОП-15 слов в РБК', fontdict={'fontweight': 'bold'}, loc='left', pad=30)
    elif media_name == 'медиазона':
        plt.title('ТОП-15 слов в Медиазоне', fontdict={'fontweight': 'bold'}, loc='left', pad=30)
    elif media_name == 'риа':
        plt.title('ТОП-15 слов в РИА новостях', fontdict={'fontweight': 'bold'}, loc='left', pad=30)
    elif media_name == 'лента':
        plt.title('ТОП-15 слов в Ленте.ru', fontdict={'fontweight': 'bold'}, loc='left', pad=30)

    plt.annotate('source: telegram-канал "Слово дня" (t.me/novosti_slovo_dnya)', (0, 0), (-10, -50), fontsize=16,
                 xycoords='axes fraction', textcoords='offset points', va='bottom')

    fig.savefig("C:/Users/alfyn/ВКР/" + file_name + '.png')

    return graph


words_top_tass = words_top(processed_words_tass)
words_top_meduza = words_top(processed_words_meduza)
words_top_interfax = words_top(processed_words_interfax)
words_top_ria = words_top(processed_words_ria)
words_top_mediazona = words_top(processed_words_mediazona)
words_top_lenta = words_top(processed_words_lenta)
words_top_rbc = words_top(processed_words_rbc)


@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    d = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    user_id = str(time.mktime(datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timetuple())).replace('.0', '')

    if call.data == "graph_1": #call.data это callback_data, которую мы указали при объявлении кнопки
        w = bot.send_message(call.message.chat.id, 'Введите интересующее вас слово: ')
        bot.register_next_step_handler(w, gr_send)

    elif call.data == "graph_2":
        keyboard_2 = types.InlineKeyboardMarkup()
        key_media_1 = types.InlineKeyboardButton(text='ТACC', callback_data='graph_top_1')
        keyboard_2.add(key_media_1)
        key_media_2 = types.InlineKeyboardButton(text='Интерфакс', callback_data='graph_top_2')
        keyboard_2.add(key_media_2)
        key_media_3 = types.InlineKeyboardButton(text='Медуза', callback_data='graph_top_3')
        keyboard_2.add(key_media_3)
        key_media_4 = types.InlineKeyboardButton(text='РИА новости', callback_data='graph_top_4')
        keyboard_2.add(key_media_4)
        key_media_5 = types.InlineKeyboardButton(text='РБК', callback_data='graph_top_5')
        keyboard_2.add(key_media_5)
        key_media_6 = types.InlineKeyboardButton(text='Лента.ru', callback_data='graph_top_6')
        keyboard_2.add(key_media_6)
        bot.send_message(call.message.chat.id, 'Выберите СМИ: ', reply_markup=keyboard_2)

    elif call.data == "graph_3":
        if len(processed_words_all) == 0:
            return
        graph_of_top(processed_words_all, 'hotpink', user_id, 'все')
        img_top = open('C:/Users/alfyn/ВКР/' + user_id + '.png', 'rb')
        bot.send_message(call.message.chat.id, 'График показывает ТОП-15 слов в новостях шести СМИ')
        bot.send_photo(call.message.chat.id, img_top)
        img_top.close()

    elif call.data == "graph_top_1":
        if len(processed_words_tass) == 0:
            return
        graph_of_top(processed_words_tass, 'deepskyblue', user_id, 'тасс')
        img_top = open('C:/Users/alfyn/ВКР/' + user_id + '.png', 'rb')
        bot.send_message(call.message.chat.id, 'График показывает ТОП-15 слов в ТАСС')
        bot.send_photo(call.message.chat.id, img_top)
        img_top.close()

    elif call.data == "graph_top_2":
        if len(processed_words_interfax) == 0:
            return
        graph_of_top(processed_words_interfax, 'deepskyblue', user_id, 'интерфакс')
        img_top = open('C:/Users/alfyn/ВКР/' + user_id + '.png', 'rb')
        bot.send_message(call.message.chat.id, 'График показывает ТОП-15 слов в Интерфаксе')
        bot.send_photo(call.message.chat.id, img_top)
        img_top.close()

    elif call.data == "graph_top_3":
        if len(processed_words_meduza) == 0:
            return
        graph_of_top(processed_words_meduza, 'palegreen', user_id, 'медуза')
        img_top = open('C:/Users/alfyn/ВКР/' + user_id + '.png', 'rb')
        bot.send_message(call.message.chat.id, 'График показывает ТОП-15 слов в Медузе (является иностранным агентом)')
        bot.send_photo(call.message.chat.id, img_top)
        img_top.close()

    elif call.data == "graph_top_4":
        if len(processed_words_ria) == 0:
            return
        graph_of_top(processed_words_ria, 'palevioletred', user_id, 'риа')
        img_top = open('C:/Users/alfyn/ВКР/' + user_id + '.png', 'rb')
        bot.send_message(call.message.chat.id, 'График показывает ТОП-15 слов в РИА новостях')
        bot.send_photo(call.message.chat.id, img_top)
        img_top.close()

    elif call.data == "graph_top_5":
        if len(processed_words_rbc) == 0:
            return
        graph_of_top(processed_words_rbc, 'palevioletred', user_id, 'рбк')
        img_top = open('C:/Users/alfyn/ВКР/' + user_id + '.png', 'rb')
        bot.send_message(call.message.chat.id, 'График показывает ТОП-15 слов в РБК')
        bot.send_photo(call.message.chat.id, img_top)
        img_top.close()

    elif call.data == "graph_top_6":
        if len(processed_words_lenta) == 0:
            return
        graph_of_top(processed_words_lenta, 'palevioletred', user_id, 'лента')
        img_top = open('C:/Users/alfyn/ВКР/' + user_id + '.png', 'rb')
        bot.send_message(call.message.chat.id, 'График показывает ТОП-15 слов в Ленте.ru')
        bot.send_photo(call.message.chat.id, img_top)
        img_top.close()


    #bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text="Какой график вы хотите увидеть?",
    #                      reply_markup=None)

@bot.message_handler(content_types=['text', 'photo'])
def gr_send(message):
    d = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    user_id = str(time.mktime(datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timetuple())).replace('.0', '')

    word = message.text
    f = freq_graph('lightskyblue', word.lower(), user_id)
    if type(f) != str:
        img = open('C:/Users/alfyn/ВКР/' + user_id + '.png', 'rb')
        bot.send_message(message.chat.id, 'График показывает количество упоминаний заданного вами слова на одну тысячу слов')
        bot.send_photo(message.chat.id, img)
        img.close()
        img_2 = open('C:/Users/alfyn/Downloads/' + user_id + '.png', 'rb')
        bot.send_message(message.chat.id, 'А в этой таблице вы можете увидеть абсолютное количество употреблений')
        bot.send_photo(message.chat.id, img_2)
        img_2.close()
    else:
        bot.send_message(message.chat.id, f)

@bot.message_handler(content_types=['text'])
def saving_feedback(message):
    message = message.text
    sqlite_connection = sqlite3.connect('C:/Users/alfyn/ВКР/feedback.db')
    cursor = sqlite_connection.cursor()
    print('База данных подключена к SQLite')

    cursor.execute('INSERT INTO feedback (feedback_text, feedback_time) VALUES(?, ?)', [message, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    sqlite_connection.commit()
    print('Изменения сохранены')

    sqlite_connection.close()
    print('База данных закрыта')
    # bot.send_message(message.chat.id, "Отзыв принят! Спасибо!")


bot.polling(none_stop=True, interval=0)

