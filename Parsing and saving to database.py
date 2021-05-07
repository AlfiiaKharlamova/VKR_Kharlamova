#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
from time import sleep
from itertools import groupby
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import pandas as pd

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


# In[2]:


tass_link = 'https://tass.ru'
tass_search = '/11'

meduza_link = 'https://meduza.io/'
meduza_search = '2021'

lenta_link = 'https://lenta.ru/'
lenta_search = 'news'

interfax_link = 'https://www.interfax.ru'
interfax_search = '/7'


# In[3]:


def parsing_urls(link, search_element):
    
    """Parsing links from news agencies front page
    Param link: Media main page link
    Param search_element: A piece of link that is distinctive for the news (needs to be updated periodically)
    """
    
    page = requests.get(link)
    soup = BeautifulSoup(page.text)
    find = soup.findAll('a') # looking for all links
    
    urls = []
    for l in find:
        variable = str(l.get('href')) # links to string
        if search_element in variable and 'goto.asp?' not in variable and 'presscenter' not in variable and 'pressreleases' not in variable and 'interview' not in variable and 'story' not in variable:
            urls.append(l.get('href'))
        sleep(0.5) # delay to avoid disconnection
    new_urls = [el for el, _ in groupby(urls)] # to remove duplicates
    
    full_urls = [link + x for x in new_urls]
    
    j = 0
    while j < len(full_urls): # loop to avoid ConnectionError while parsing Lenta headlines and news
        if 'ru/https' in full_urls[j] or '/https:' in full_urls[j] or 'ruhttps' in full_urls[j]:
            del full_urls[j]
        else:
            j = j + 1
            
    return full_urls


# In[4]:


def parsing_urls_ria():
    
    page = requests.get('https://ria.ru/')
    soup = BeautifulSoup(page.text)
    find = soup.findAll('a') # looking for all links
    
    urls = []
    for l in find:
        variable = str(l.get('href')) # links to string
        if '20' in variable:
            urls.append(l.get('href'))
        sleep(0.5) # delay to avoid disconnection
    new_urls = [el for el, _ in groupby(urls)] # to remove duplicates
    
    full_urls = ['https://ria.ru/' + x for x in new_urls]
    ria_full_urls = []
    
    for j in full_urls: #loop to avoid dubble url adress in ria
        j = j.replace('https://ria.ru/', '', 1)
        ria_full_urls.append(j)
        
#     'https://' not in ria_full_urls[j]
    
    j = 0
    while j < len(ria_full_urls): # loop to avoid ConnectionError while parsing Lenta headlines and news
        if 'https://' not in ria_full_urls[j] or 'ru/https' in ria_full_urls[j] or '/https:' in ria_full_urls[j] or 'ruhttps' in ria_full_urls[j] or 'mama.ria.ru' in ria_full_urls[j] or 'euro2020' in ria_full_urls[j]:
            del ria_full_urls[j]
        else:
            j = j + 1
            
    return ria_full_urls


# In[5]:


def parsing_urls_mediazona():
    page = requests.get('https://zona.media/news#more')
    soup = BeautifulSoup(page.content)
      
    find = soup.findAll('a', {'class': 'mz-topnews-item__link-wrapper'})

    urls = []
    for l in find:
        variable = str(l.get('href')) # links to string
        if '2021' in variable:
            urls.append(l.get('href'))
        sleep(0.5) # delay to avoid disconnection
    new_urls = [el for el, _ in groupby(urls)] # to remove duplicates
    
    full_urls = ['https://zona.media' + x for x in new_urls]
    
    return full_urls


# In[7]:


def parsing_urls_rbc():
    page = requests.get('https://www.rbc.ru/')
    soup = BeautifulSoup(page.content)
      
    find = soup.find('div', {'class': 'js-news-feed-list'}).findAll('a')
    urls = []
    for l in find:
        variable = str(l.get('href')) # links to string
        if 'from' in variable:
            urls.append(variable)

        sleep(0.5) # delay to avoid disconnection

    
    return urls


# In[160]:


# def parsing_urls_interfax():
#     page = requests.get('https://www.interfax.ru/')
#     soup = BeautifulSoup(page.content)
      
#     find = soup.findAll('a')

#     urls = []
#     for l in find:
#         variable = str(l.get('href')) # links to string
#         if '7' in variable and 'goto.asp?' not in variable and 'presscenter' not in variable and 'pressreleases' not in variable and 'interview' not in variable and 'story' not in variable:
#             urls.append(l.get('href'))
#         sleep(0.5) # delay to avoid disconnection
#     new_urls = [el for el, _ in groupby(urls)] # to remove duplicates
    
#     full_urls = ['https://www.interfax.ru' + x for x in new_urls]
    
#     j = 0
#     while j < len(full_urls): # loop to avoid ConnectionError while parsing Lenta headlines and news
#         if 'ru/https' in full_urls[j] or '/https:' in full_urls[j] or 'ruhttps' in full_urls[j] or 'utm_source=ed_ch' in full_urls[j]:
#             del full_urls[j]
#         j = j + 1
                
#     return full_urls


# In[6]:


tass_links = parsing_urls(tass_link, tass_search)
meduza_links = parsing_urls(meduza_link, meduza_search)
ria_links = parsing_urls_ria()
mediazona_links = parsing_urls_mediazona()
lenta_links = parsing_urls(lenta_link, lenta_search)
rbc_links = parsing_urls_rbc()
interfax_links = parsing_urls(interfax_link, interfax_search)


# In[9]:


def parsing_tass_into_table(urls):
    
    full_table = []
    
    j = 0
    
    while j < len(urls):
        one_record = []
        
        p = requests.get(urls[j])
        s = BeautifulSoup(p.content)
        
        one_record.append(urls[j])
        
        media = 1
        one_record.append(media)
        
        headline = s.findAll('meta', {'property':'og:title'})
        head = ''
        for i in headline:
            head = head + i['content']
        one_record.append(head)
        
        text = s.findAll('p', {'class': None})
        text_full = ''
        for i in text:
            text_full = text_full + i.text.replace('\xa0', ' ') + ' '
        one_record.append(text_full)
        
        unixdate = s.find('dateformat')['time']
        one_record.append(int(unixdate))
        
        tags = s.findAll('a', {'class':'tags__item'})
        tags_list = ''
        for i in tags:
            tags_list = tags_list + i.text + '; '
        one_record.append(tags_list)
        
        full_table.append(one_record)
        
        j = j + 1
        sleep(0.5)
    
    return full_table


# In[10]:


def parsing_meduza_into_table(urls):
    
    full_table = []
    
    j = 0
    
    while j < len(urls):
        one_record = []
        
        p = requests.get(urls[j])
        s = BeautifulSoup(p.content)
        
        one_record.append(urls[j])
        
        media = 2
        one_record.append(media)
        
        headline = s.findAll('meta', {'property':'og:title'})
        head = ''
        for i in headline:
            head = head + i['content']
        one_record.append(head)
        
        text = s.findAll('p', {'class': 'SimpleBlock-module_p__Q3azD'})
        text_full = ''
        for i in text:
            text_full = text_full + i.text.replace('\xa0', ' ') + ' '
        one_record.append(text_full)
        
        date = s.find('time')
        if date != None:
            date = date.text.replace(',','').replace('января','01').replace('февраля','02').replace('марта','03').replace('апреля','04').replace('мая','05').replace('июня','06').replace('июля','07').replace('августа','08').replace('сентября','09').replace('октября','10').replace('ноября','11').replace('декабря','12')
            unixdate = time.mktime(datetime.strptime(date, "%H:%M %d %m %Y").timetuple())
        one_record.append(int(unixdate))
        
        tags = 'null'
        one_record.append(tags)

        full_table.append(one_record)
        
        j = j + 1
        sleep(0.5)
    
    return full_table


# In[11]:


def parsing_interfax_into_table(urls):
    
    full_table = []
    
    j = 0
    
    while j < len(urls):
        one_record = []
        
        p = requests.get(urls[j])
        s = BeautifulSoup(p.content)
        
        one_record.append(urls[j])
        
        media = 3
        one_record.append(media)
        
        headline = s.findAll('meta', {'property':'og:title'})
        head = ''
        for i in headline:
            head = head + i['content']
        one_record.append(head)
        
        text = s.findAll('p', {'class': None})
        text_full = ''
        for i in text:
            text_full =  text_full + i.text.replace('\xa0', ' ') + ' '
        one_record.append(text_full)
        
        date = s.find('time')
        if date != None:
            date = date.text.replace('\n','').replace('\r','').replace('\t','').replace(',','').replace('января','01').replace('февраля','02').replace('марта','03').replace('апреля','04').replace('мая','05').replace('июня','06').replace('июля','07').replace('августа','08').replace('сентября','09').replace('октября','10').replace('ноября','11').replace('декабря','12')
            unixdate = time.mktime(datetime.strptime(date, "%H:%M %d %m %Y").timetuple())
        one_record.append(int(unixdate))
        
        
        tags = s.find('div', {'class':'textMTags'})
        if tags != None:
            tags = tags.findAll('a')
            tags_list = ''
            for i in tags:
                tags_list = tags_list + i.text + '; '
        one_record.append(tags_list)
        
        full_table.append(one_record)
        
        j = j + 1
        sleep(0.5)
        
    return full_table


# In[22]:


def parsing_ria_into_table(urls):
    
    full_table = []
    
    j = 0
    
    while j < len(urls):
        one_record = []
        
        p = requests.get(urls[j])
        s = BeautifulSoup(p.content)
        
        one_record.append(urls[j])
        
        media = 4
        one_record.append(media)
        
        headline = s.findAll('h1', {'class':'article__title'})
        head = ''
        for i in headline:
            head = head + i.text
        one_record.append(head)
        
        text = s.findAll('div', {'class': 'article__text'})
        text_full = ''
        for i in text:
            text_full = text_full + i.text.replace('\xa0', ' ') + ' '
        one_record.append(text_full)
        
        date = s.find('div', {'class':'article__info-date'})
        if date != None:
            date = date.find('a').text
            unixdate = time.mktime(datetime.strptime(date, "%H:%M %d.%m.%Y").timetuple())
        one_record.append(int(unixdate))
                
        tags = s.findAll('a', {'class': 'article__tags-item'})
        tags_list = ''
        for i in tags:
            tags_list = tags_list + i.text + '; '
        one_record.append(tags_list)
        
        full_table.append(one_record)
                
        j = j + 1
        sleep(0.5)
    
    for i in full_table:
        if len(i) > 4:
            if 'Найди меня, мама; Жизнь без преград' in i[3]:
                full_table.remove(i)

    return full_table


# In[13]:


def parsing_mediazona_into_table(urls):
    
    full_table = []
    
    j = 0
    
    while j < len(urls):
        one_record = []
        
        p = requests.get(urls[j])
        s = BeautifulSoup(p.content)
        
        one_record.append(urls[j])
        
        media = 5
        one_record.append(media)
        
        head = s.find('header', {'class':'mz-publish__title'})
        if head != None:
            head = head.text.replace('\xa0', ' ')
        one_record.append(head)
        
        text = s.findAll('p')
        text_full = ''
        for i in text:
            text_full = text_full + i.text.replace('\xa0', ' ') + ' '
        one_record.append(text_full)
        
        date = s.find('section', {'class':'mz-publish-meta'})
        if date != None:
            date = date.findAll('div')[1].text.replace(',','').replace('января','01').replace('февраля','02').replace('марта','03').replace('апреля','04').replace('мая','05').replace('июня','06').replace('июля','07').replace('августа','08').replace('сентября','09').replace('октября','10').replace('ноября','11').replace('декабря','12')
            unixdate = time.mktime(datetime.strptime(date, "%d %m %Y %H:%M").timetuple())
        one_record.append(int(unixdate))
        
        tags = 'null'
        one_record.append(tags)
        
        full_table.append(one_record)
        
        j = j + 1
        sleep(0.5)
    
    return full_table


# In[14]:


def parsing_lenta_into_table(urls):
    
    full_table = []
    
    j = 0
    
    while j < len(urls):
        one_record = []
        
        p = requests.get(urls[j])
        s = BeautifulSoup(p.content)
        
        one_record.append(urls[j])
        
        media = 6
        one_record.append(media)
        
        head = s.find('h1')
        if head != None:
            head = head.text.replace('\xa0', ' ')
        one_record.append(head)
        
        text = s.findAll('p')
        text_full = ''
        for i in text:
            text_full = text_full + i.text.replace('\xa0', ' ') + ' '
        one_record.append(text_full)
        
        date = s.find('time', {'class':'g-date'})
        if date != None:
            date = date['datetime'].replace('T', ' ')
            date = date[:-9]
            unixdate = time.mktime(datetime.strptime(date, '%Y-%m-%d %H:%M').timetuple())
        one_record.append(int(unixdate))
        
        tags = 'null'
        one_record.append(tags)
        
        full_table.append(one_record)
        
        j = j + 1
        sleep(0.5)
    
    return full_table


# In[15]:


def parsing_rbc_into_table(urls):
    
    full_table = []
    
    j = 0
    
    while j < len(urls):
        one_record = []
        
        p = requests.get(urls[j])
        s = BeautifulSoup(p.content)
        
        one_record.append(urls[j])
        
        media = 7
        one_record.append(media)
        
        head = s.find('h1', {'class':'article__header__title-in js-slide-title'})
        if head != None:
            head = head.text.replace('\xa0', ' ').replace('\n', ' ').replace('\r', ' ')
        one_record.append(head)
        
        text = s.findAll('p')
        text_full = ''
        for i in text:
            text_full = text_full + i.text.replace('\xa0', ' ').replace('\n', ' ').replace('\r', ' ') + ' '
        one_record.append(text_full)
        
        date = s.find('span', {'class':'article__header__date'}) 
        if date != None:
            date = date['content'].replace('T', ' ')
            date = date[:-9]
            unixdate = time.mktime(datetime.strptime(date, '%Y-%m-%d %H:%M').timetuple())
            one_record.append(int(unixdate))
        else:
            one_record.append(None)
    
        tags = s.findAll('a', {'class': 'article__tags__item'})
        tags_list = ''
        for i in tags:
            tags_list = tags_list + i.text.replace('\n', ' ').replace('\r', ' ') + '; '
        one_record.append(tags_list)

        full_table.append(one_record)
        
        j = j + 1
        sleep(0.5)
    
    return full_table


# In[17]:


tass_news = parsing_tass_into_table(tass_links)
meduza_news = parsing_meduza_into_table(meduza_links)
mediazona_news = parsing_mediazona_into_table(mediazona_links)
ria_news = parsing_ria_into_table(ria_links)
interfax_news = parsing_interfax_into_table(interfax_links)
lenta_news = parsing_lenta_into_table(lenta_links)
rbc_news = parsing_rbc_into_table(rbc_links)


# In[29]:


def saving_data_to_db(parsed_data):
    cursor.executemany('INSERT OR IGNORE INTO media_news (news_id, media_id, headline, text, unixdate, tags) VALUES (?,?,?,?,?,?)', parsed_data)
    print('Insert done')


# In[30]:


sqlite_connection = sqlite3.connect('C:/Users/alfyn/ВКР/sqlite_python.db')
print('База данных подключена к SQLite')
cursor = sqlite_connection.cursor()

saving_data_to_db(tass_news)
saving_data_to_db(meduza_news)
saving_data_to_db(mediazona_news)
saving_data_to_db(ria_news)
saving_data_to_db(interfax_news)
saving_data_to_db(lenta_news)
saving_data_to_db(rbc_news)

sqlite_connection.commit()
print('Изменения сохранены')
    
sqlite_connection.close()    
print('База данных закрыта')

