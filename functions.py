# Файл для хранения функций для ноутбука третьего этапа соревнований от Моторики
# Импортируем библиотеки
import pandas as pd
import numpy as np

# графические библиотеки
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# библиотеки машинного обучения
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# отображать по умолчанию длину Датафрейма
pd.set_option("display.max_rows", 9, "display.max_columns", 9)

# библиотека взаимодействия с интерпретатором
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import os


gestures = ['"open"',  # 0
            '"пистолет"',  # 1
            'сгиб большого пальца',  # 2
            '"ok"',  # 3
            '"grab"',  # 4
            '"битые" данные',  # -1
]




def privet(name):   # print 'privet' to a given name
    print(f'privet {name}')

def get_sensor_list(Pilot_id, X_train, print_active=False):
    """
    Функция печати и импорта в память всех номеров датчиков
    Аргументы функции:
    Pilot_id - номер пилота,
    X_train - обучающая выборка. 
    """
    
    df = pd.DataFrame(data = X_train[Pilot_id], index = [s for s in range(X_train[Pilot_id].shape[0])], 
                        columns = [s for s in range(X_train[Pilot_id].shape[1])]
    ).T
    
    # Создадим список индексов активных и пассивных датчиков. Среднее значение сигнала не превышает 200 единиц.
    active_sensors, passive_sensors = list(), list()
    #reliable_sensors, unreliable_sensors =  list(), list()
    
    for i in range(df.shape[0]):
        # если средняя амплитуда превышает 200, то добавляем индекс в 'active_sensors'
        if df.iloc[i].mean() > 200:
            active_sensors.append(i)
                   
            # Если разница между абсолютными средними значениями за последние 15 сек и первые 60 сек превышает 200,
            # то датчики заносим в список надежных. Остальные датчики с малой амплитудой - в список ненадёжных. 
        #     if abs(df.iloc[i][0:49].mean() - df.iloc[i][85:].mean()) > 200:
        #         reliable_sensors.append(i)
        #     else:
        #         unreliable_sensors.append(i)
        else:
            passive_sensors.append(i)

    if print_active is True:
        print(f"Активные датчики пилота " + str(Pilot_id) + ": ", active_sensors)
        print(f"Пассивные датчики пилота " + str(Pilot_id) + ": ", passive_sensors)
    #elif print_reliable is True:
    #    print(f"Датчики с большой амплитудой, наблюдения " + str(id) +": ", reliable_sensors)
    #    print(f"Датчики с малой амплитудой, " + str(id) +": ", unreliable_sensors)  
    
    return active_sensors, passive_sensors #, reliable_sensors, unreliable_sensors



def get_all_sensors_plot(Pilot_id, timesteps:list, X_train, plot_counter=1):
    """
    Функция построения диаграммы показаний датчиков заданного временного периода. Аргументы функции:
    Pilot_id - номер пилота;
    timesteps - временной период;
    X_train - обучающая выборка;
    plot_counter - порядковый номер рисунка.
    """
    
    df = pd.DataFrame(data = X_train[Pilot_id], index = [s for s in range(X_train[Pilot_id].shape[0])], 
                        columns = [s for s in range(X_train[Pilot_id].shape[1])]
    )
    
    fig = px.line(data_frame=df.iloc[timesteps[0]:timesteps[1],:])
    
    fig.update_layout(
        title=dict(text=f'Рис. {plot_counter}'+' - наблюдение ' + str(Pilot_id), x=.5, y=0.05, xanchor='center'), 
        xaxis_title_text = 'Время, сек', yaxis_title_text = 'Показатель', # yaxis_range = [0, 3000],
        legend_title_text='Индекс <br>датчика',
        width=600, height=400,
        margin=dict(l=100, r=60, t=80, b=100),
    )

    fig.show()

    # сохраним результат в папке figures. Если такой папки нет, то создадим её
    if not os.path.exists("figures"):
        os.mkdir("figures")

    #fig.write_image(f'figures/fig_{plot_counter}.png', engine="kaleido")


def get_active_passive_sensors_plot(Pilot_id, X_train, time_start=0, time_end=500, plot_counter=1):
    """
    Функция построения графика показаний активных и пассивных датчиков.
    Аргумент функции:
    id - номер наблюдения;
    X_train - обучающая выборка;
    plot_counter - порядковый номер рисунка.  
    """
    # списки сенсоров
    active_sensors, passive_sensors = get_sensor_list(Pilot_id, X_train)  #, print_active=True

    timesteps=[time_start, time_end]

    df = pd.DataFrame(data = X_train[Pilot_id], index = [s for s in range(X_train[Pilot_id].shape[0])], 
                        columns = [s for s in range(X_train[Pilot_id].shape[1])]
    ).iloc[timesteps[0]:timesteps[1],:]
    
        
    df_1 = pd.DataFrame(df[active_sensors], columns=active_sensors)
    df_2 = pd.DataFrame(df[passive_sensors], columns=passive_sensors)

   
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('активные датчики', 'пассивные датчики')
    )
    
    for i in df_1.columns: fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i], name=str(df[i].name)), row=1, col=1)

    for i in df_2.columns: fig.add_trace(go.Scatter(x=df_2.index, y=df_2[i], name=str(df[i].name)), row=1, col=2)

    fig.update_layout(title={'text':f'Рис. {plot_counter}'+' - Активные и пассивные датчики пилота ' + str(Pilot_id), 'x':0.5, 'y':0.05}
    )

    fig.update_layout(width=1000, height=400, legend_title_text ='Номер датчика',
                        xaxis_title_text  = 'Время',  yaxis_title_text = 'Сигнал датчика', yaxis_range=  [0, 3500], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Сигнал датчика', yaxis2_range= [0 , 200],
                        margin=dict(l=100, r=60, t=80, b=100), 
                        #showlegend=False # легенда загромождает картинку
    )

    fig.show()

    #fig.write_image(f'figures/fig_{plot_counter}.png', engine="kaleido")

def plot_history(history):
    
    """
    Функция визуализации процесса обучения модели       
    """
    f1_sc = history.history['f1']  
    loss = history.history['loss']

    f1_sc_val = history.history['val_f1'] # на валидационной выборке
    val_loss = history.history['val_loss']

    epochs = range(len(f1_sc))

    # визуализация систем координат
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 4))

    ax[0].plot(epochs, loss, 'b', label='Training loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[0].set_title('Качество обучения модели')
    ax[0].legend()

    ax[1].plot(epochs, f1_sc, 'b', label='Training f1_score')
    ax[1].plot(epochs, f1_sc_val, 'r', label='Training val_f1_score')
    ax[1].set_title('Изменение f1_score') # изменение f1-score
    ax[1].legend()

    #fig.show()

def get_gesture_prediction_plot(id_pilot, i, y_pred_train_nn_mean, mounts, plot_counter):
    """
    Функция построения графиков: сигнал датчиков оптомиографии, изменение класса жеста, вероятности появления жеста и предсказание класса жеста.
    Агументы:
    id_pilot = 3  # номер пилота
    plot_counter = 1 # номер рисунка
    i - номер наблюдениия
    mounts - словарь с данными
    """
    
    mount = mounts[id_pilot]         # выбираем номер пилота
    X_train_nn = mount['X_train_nn']
    y_train_nn = mount['y_train_nn']
    #y_pred_train_nn = mount['y_pred_train_nn']
    #y_pred_train_nn_mean = np.mean(x_trn_pred_dict[id_pilot], axis=0)

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    plt.suptitle(f'Рис. {plot_counter} - наблюдение {i} пилота {id_pilot}' , y=-0.01, fontsize=16)
    
    plt.subplots_adjust(  left=0.1,   right=0.9,
                        bottom=0.1,     top=0.9,
                        wspace=0.1,  hspace=0.4)
 
    ax[0].plot(X_train_nn[i])
    ax[0].set_title('Сигналы датчиков ОМГ')

    ax[1].imshow(y_train_nn[i].T, origin="lower")
    ax[1].set_aspect('auto')
    ax[1].set_title('Класс / жест')
    ax[1].set_yticks(
        np.arange(5),
        ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    )

    ax[2].imshow(y_pred_train_nn_mean[i].T, origin="lower")
    ax[2].set_aspect('auto')
    ax[2].set_title('Предсказание вероятностей появления классов жестов')
    ax[2].set_yticks(
        np.arange(5),
        ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    )

    ax[3].plot(y_pred_train_nn_mean[i].argmax(axis=-1))
    ax[3].set_aspect('auto')
    ax[3].set_title('Предсказание классов жестов')
    ax[3].set_yticks(
        np.arange(5),
        ['Open', 'Pistol', 'Thumb', 'OK', 'Grab']
    )
    ax[3].set_xlabel('Время')
    plt.tight_layout()


#Pilot_id = 1
#time_start = 100
#time_end = 600
#sensors = list([0, 2, #5, 8, 12, 15, 17, 19, 21, 24, 27, 29, 30, 33, 36, 38
#])

#plot_counter = 6




def get_signal_derivative_plot(Pilot_id, timesteps:list, sensors:list, mounts, plot_counter=1):
    """
    Функция построения графиков: сигнал датчиков оптомиографии, изменение класса жеста, вероятности появления жеста и предсказание
    Агументы:
    id_pilot = 3  # номер пилота
    plot_counter = 1 # номер рисунка
    i - номер наблюдениия
    mounts - словарь с данными   
    """
    X_train=mounts[Pilot_id]['X_train']
    y_train=mounts[Pilot_id]['y_train']

    
    df_1 = pd.DataFrame(
        data = X_train, 
        index = [s for s in range(X_train.shape[0])], 
        columns = [s for s in range(X_train.shape[1])]
    ).iloc[timesteps[0]:timesteps[1],:][sensors]


    df_2 = pd.DataFrame(
        data = y_train, 
        index = [s for s in range(y_train.shape[0])]
    ).iloc[timesteps[0]:timesteps[1],:]

    
    # Нормализация данных
    scaler = StandardScaler()
    scaler.fit(df_1)
    df_1 = pd.DataFrame(scaler.transform(df_1))

    fig = make_subplots(rows=2, cols=2, 
        subplot_titles=(f'X_train - сигналы датчиков', f'Производная сигналов датчиков',
        'y_train', f'Квадрат производной сигналов датчиков'), vertical_spacing = 0.1,
    )

    for i in df_1.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i], name=str(df_1[i].name)), row=1, col=1)
    

    for i in df_2.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_2[i], name=str(df_1[i].name)), row=2, col=1)


    df_3 = pd.DataFrame(df_1.diff(), index = range(df_1.index[0], df_1.index[-1]+1))


    for i in df_3.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i], name=str(df_1[i].name)), row=1, col=2)

    #  датасет квадрата производной
    df_4 = pd.DataFrame(np.power(df_1.diff(),2), index = range(df_1.index[0], df_1.index[-1]+1))

    for i in df_4.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_4[i], name=str(df_4[i].name)), row=2, col=2)

    fig.update_layout(title={'text':f'Рис. {plot_counter} - Нормализованные сигналы датчиков {sensors} пилота {Pilot_id}', 
    'x':0.5, 'y':0.01}
    )

    fig.update_layout(width=1200, height=800, legend_title_text ='Номер датчика',
                        xaxis_title_text  = 'Время',  yaxis_title_text = 'Сигнал датчика', #yaxis_range=[1500, 1700], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Жест', #yaxis2_range= [0 , 200],
                        margin=dict(l=40, r=60, t=30, b=80), 
                        showlegend=False # легенда загромождает картинку
    )

    fig.show()


def get_normalized_signal_derivative_plot(Pilot_id, timesteps:list, sensors:list, mounts,
                               plot_counter=1):

    """
    
    """
    X_train=mounts[Pilot_id]['X_train']
    y_train=mounts[Pilot_id]['y_train']
    
    Pilot_id = 1
    time_start = 100
    time_end = 1450
    sensors = list([0, 2, 5, 8, 12, 15, 17, 19, 21, 24, 27, 29, 30, 33, 36, 38
    ])

    plot_counter = 7

    timesteps=[time_start, time_end]

    fig = make_subplots(rows=4, cols=2, 
                        subplot_titles=(
                        f'Исходные сигналы датчиков', f'Нормализованные сигналы датчиков',
                        'y_train', 'y_train',
                        f'Производная сигналов датчиков', f'Производная нормализованных сигналов датчиков', 
                        f'Квадрат производной сигналов датчиков', f'Квадрат производной нормализованных сигналов датчиков'
                        ), vertical_spacing = 0.1,
    )

    df_1 = pd.DataFrame(
        data = X_train[Pilot_id], 
        index = [s for s in range(X_train[Pilot_id].shape[0])], 
        columns = [s for s in range(X_train[Pilot_id].shape[1])]
        ).iloc[timesteps[0]:timesteps[1],:][sensors]

    for i in df_1.columns: 
        fig.add_trace(go.Scatter(x=df_1.index, y=df_1[i], name=str(df[i].name)), row=1, col=1)

    df_2 = pd.DataFrame(data = y_train[Pilot_id], index = [s for s in range(y_train[Pilot_id].shape[0])]).iloc[timesteps[0]:timesteps[1],:]

    for i in df_2.columns: 
        fig.add_trace(go.Scatter(x=df_2.index, y=df_2[i], name=str(df[i].name)), row=2, col=1)

    df_3 = pd.DataFrame(df_1.diff(), index = range(df_1.index[0], df_1.index[-1]+1))

    for i in df_3.columns: 
        fig.add_trace(go.Scatter(x=df_3.index, y=df_3[i], name=str(df[i].name)), row=3, col=1)

    #  датасет квадрата производной
    df_4 = pd.DataFrame(np.power(df_1.diff(),2), index = range(df_1.index[0], df_1.index[-1]+1))

    for i in df_4.columns: 
        fig.add_trace(go.Scatter(x=df_4.index, y=df_4[i], name=str(df[i].name)), row=4, col=1)

    # Нормализация данных
    scaler = StandardScaler()
    scaler.fit(df_1)
    df_5 = pd.DataFrame(scaler.transform(df_1))

    for i in df_5.columns: 
        fig.add_trace(go.Scatter(x=df_5.index, y=df_5[i], name=str(df_5[i].name)), row=1, col=2)

    #df_6 = pd.DataFrame(data = y_train[Pilot_id], index = [s for s in range(y_train[Pilot_id].shape[0])]).iloc[timesteps[0]:timesteps[1],:]

    for i in df_2.columns: 
        fig.add_trace(go.Scatter(x=df_2.index, y=df_2[i], name=str(df[i].name)), row=2, col=2)


    df_7 = pd.DataFrame(df_5.diff(), index = range(df_5.index[0], df_5.index[-1]+1))


    for i in df_7.columns: 
        fig.add_trace(go.Scatter(x=df_5.index, y=df_7[i], name=str(df_5[i].name)), row=3, col=2)

    #  датасет квадрата производной
    df_8 = pd.DataFrame(np.power(df_5.diff(),2), index = range(df_5.index[0], df_5.index[-1]+1))

    for i in df_8.columns: 
        fig.add_trace(go.Scatter(x=df_5.index, y=df_8[i], name=str(df_5[i].name)), row=4, col=2)

    fig.update_layout(title={'text':f'Рис. {plot_counter} - Преобразование сигнала датчиков {sensors} пилота {Pilot_id}', 'x':0.5, 'y':0.01}
    )

    fig.update_layout(width=1200, height=1200, legend_title_text =f'Номер датчика ',
                        xaxis_title_text  = 'Время',  yaxis_title_text = 'Сигнал датчика', #yaxis_range=[0, 4000], 
                        xaxis2_title_text = 'Время', yaxis2_title_text = 'Жест', #yaxis2_range= [0 , 8],
                        xaxis3_title_text = 'Время', yaxis3_title_text = 'Жест', #yaxis3_range= [-1 , 8],
                        xaxis4_title_text = 'Время', yaxis4_title_text = 'Жест', #yaxis2_range= [0 , 8],
                        margin=dict(l=40, r=60, t=30, b=80), 
                        showlegend=False # легенда загромождает картинку
    )

    fig.show()