Добро пожаловать в открытый инструмент для парсинга новостых текстов и визуализации полученных данных!

Программа состоит из 4 основных компонентов: парсер(parsing_and_saving_to_db.py), 
обработчик текста(retrieve_and_preprocessing.py), создание частотных графиков(froq_plots) 
и создание графа именованных сущностей(ner_graph.py)

На данный момент пограмма берет новости семи интернет-СМИ: Медуза, ТАСС, Интерфакс, РИА новости, Медиазона, РБК, Лента.ru.
В будущем планируется расширение круга этих СМИ.

Все собранные новости сохраняются в таблицу 'media_news' базы данных 'sqlite_python.db', файл которой также приложен к проекту.
База будет периодически обновляться, однако вы можете скачать этот файл и сами его дополнять или же создать свою базу для этих целей.

Вы можете запускать алгоритмы из cmd, только не забудьте проставить необходимые аргументы во все вызываемые функции, 
которые перечислены ниже. В алгоритме парсинга можно ничего не менять, просто делайте переодически git pull, 
чтобы всегда иметь свежий код.
Алгоритм содержит следующие функции, требующие ввода кастомных параметров:
1. retrieve_by_date(unixdate, db_path) - в качестве второго параметра укажите полный путь до базы данных, 
которую вы сохранили на свой компьютер. Это можно сделать в куске, где результат работы функции созраняется в переменные:

db_col_meduza = retrieve_by_date_and_media(condition_list_meduza, 'C:/Users/sqlite_python.db')
db_col_tass = retrieve_by_date_and_media(condition_list_tass, 'C:/Users/sqlite_python.db')
db_col_interfax = retrieve_by_date_and_media(condition_list_interfax, 'C:/Users/sqlite_python.db')
db_col_ria = retrieve_by_date_and_media(condition_list_ria, 'C:/Users/sqlite_python.db')
db_col_mediazona = retrieve_by_date_and_media(condition_list_mediazona, 'C:/Users/sqlite_python.db')
db_col_lenta = retrieve_by_date_and_media(condition_list_lenta, 'C:/Users/sqlite_python.db')
db_col_rbc = retrieve_by_date_and_media(condition_list_rbc, 'C:/Users/sqlite_python.db')

2. retrieve_by_date_and_media(condition_list, db_path) - в качестве второго параметра укажите полный путь до базы данных, 
которую вы сохранили на свой компьютер
3. В коде по умолчанию берутся все новости за текущую дату. Если вы хотите получить новости за свою дату, найдите косок

condition_list_tass = getting_condition_list(1)
condition_list_meduza = getting_condition_list(2)
condition_list_interfax = getting_condition_list(3)
condition_list_ria = getting_condition_list(4)
condition_list_mediazona = getting_condition_list(5)
condition_list_lenta = getting_condition_list(6)
condition_list_rbc = getting_condition_list(7)

а затем поменяйте getting_condition_list(media) на getting_condition_list_user(media, date), где это необходимо.

4. В функцию preprocess_text(db_col) необходимо задать результат работы фукции по извлечению данных их Базы. В коде 
я сохраняю весь обработанный тест в переменные:

processed_words_tass = preprocess_text(db_col_tass)
processed_words_meduza = preprocess_text(db_col_meduza)
processed_words_interfax = preprocess_text(db_col_interfax)
processed_words_ria = preprocess_text(db_col_ria)
processed_words_mediazona = preprocess_text(db_col_mediazona)
processed_words_lenta = preprocess_text(db_col_lenta)
processed_words_rbc = preprocess_text(db_col_rbc)

5. friq_graph(color, barplot_path) - данная функция генерирует столбиковую диаграмму по употреблению конкретного слова, 
введенного пользователем (количество употреблений слова на 1000 слов), а также таблицу с абсолютным количеством употреблений. 
Необходимо указать цвет диаграммы (названия цветов даны в картинке seaborn_colors), интересующее слово, путь с названием самого 
файла, куда будет сохранен график. Обратите внимание, что таблица по умолчанию скачивается в загрузки.
6. words_top(processed_words_media) - эта функция создает словарь ТОП-15 слов в выбранном медиа,
сюда подается список обработанных слов. Обратите внимание, что данная функция используется внутри следующей функции
7. graph_of_top(processed, color) - данная функция генерирует непосредственно график ТОП-15 слов в выбранном медиа на основе
вышеуказанного словаря. Сюда необходимо подать список обработанных слов, который пробросится через предыдущую функцию, а также
необходимо указать название цвета (все цвета - в картинке seaborn_colors)

Что касается функций по извлечению данных из базы:
1. getting_current_unix() - функция генерирующая сегодняшнуюю дату в формате unix - на вход ей ничего не нужно, 
она возвращает список из двух unix datetime (первая и последняя секунды сегодняшнего дня)
2. getting_user_unix(date) - данная функция возвращает дату, указанную в качестве аргумента, введенную в формате 'YYYY-MM-DD',
возвращает такой же список, как и предыдущая функция, только с кастомной датой
3. retrieve_by_date(unixdate, db_path) - в базе даты выхода новостей храняться в формате unix, поэтому чтобы обратиться 
к записям по дате, необходимо его сгенерировать одной из вышеуказанных функций
4. retrieve_by_date_and_media(condition_list, db_path) - чтобы обратиться к дате и конкретному медиа, мы используем такой объект как 
condition_list - это список, который содержит две unix datetime, а также media_id 
(1 - tass; 2 - meduza; 3 - interfax; 4 - ria; 5 - mediazona; 6 - lenta.ru; 7 - rbc' - см.таблицу в базе 'media_list')
5. getting_condition_list(media) и getting_condition_list_user(media, date) - тот же принцип, что и у генерации даты.
Первая функция генерирует condition_list для запроса к базе на сегодняшней дате и на media_id, а вторая на введенной 
пользователем дате в формате 'YYYY-MM-DD' и media_id

Не забудьте установить все необходимые библиоткеки, которые указаны в requirements.txt

Приятной работы!