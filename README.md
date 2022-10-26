# Motorica_3
Проект реализован в последовательности ноутбуков и набора файлов:

    Файл *1_EDA_sprint_3.ipynb* представляет собой EDA с анализом тренировочных и тестовых данных до начала работы с моделью, в т.ч. анализ показаний датчиков и жестов. Используемые в ноутбуке функции приведены в файле functions.py, который должен находиться в той же папке, что и *1_EDA_sprint_3.ipynb*.

    **Возможно что-то еще нужно дописать после доработки файла**  

    Файл *2_model_nn.ipynb* является доработкой baseline, включающий в себя решение задачи с помощью двух моделей нейросети. В первой части ноутбука данные загружаются из архива и преобразуются с помощью библиотеки mne для последующей подачи в модель. Далее последовательно обучаются две модели: SimpleRNN (первая модель, в основе лежит слой SimpleRNN библиотеки Keras) и LSTM (вторая модель, в основе лежит несколько слоев LSTM библиотеки Keras и дополнительного Dense-слоя).  
    Основная задача работы первой модели - это предсказание изменения жеста (появления  "ступеньки") по данным X_train для последующего обучения более сложной модели. Использование упрощенной модели SimpleRNN совместно с функцией активации 'sigmoid' (activation='sigmoid') в выходном слое и использование loss="mean_squared_error" при сборке модели позволяет сделать так, чтобы модель предсказывала "ступеньку" по данным датчиков (X_train). Модель учитывает классы из y_train, а время начала ступеньки берет из X_train. Примечание: Изначально y_train представляет собой данные по жестам, которые определены манипулятором. При подготовке данных для обучения человек ("пилот") с набором датчиков повторяет жесты за манипулятором, в результате чего данные X_train запаздывают на некоторое время относительно исходного y_train.  
    Вторая модель обучается на X_train и измененных данных y_train_ch (предсказание обученной модели SimpleRNN на X_train). Далее обученная модель LSTM используется для предсказания тестовых данных.
    
    
    
    представляет собой решение задачи с помощью модели нейросети, в котором описаны идеи и используемые способы предобработки данных, архитектура и обучение модели реккуррентной нейросети, а также возможный способ постобработки предикта перед сабмитом. Файл лучше открывать в Colab, предварительно разместив файлы X_train.npy, X_test.npy, y_train.csv, sample_submission.csv в '/content/drive/MyDrive/Motorica/' либо изменив переменную PATH. Веса двух лучших моделей, давших примерно одинаковые значения при оценке метрики во время обучения (f1_score > 0,99) и одинаковые значения на Leaderbord приведены в файлах "3_1_best_weights_st12_9_temp.hdf5" и "3_2_best_weights_st12_13.hdf5".

    Файлы X_train.npy, X_test.npy, y_train.csv, sample_submission.csv - исходные файлы тренировочного, тестового фичей, тренировочного таргета и файла примера загрузки на Kaggle предсказанных данных. Если файлы ноутбука открываются через Colab, то вышеуказанные файлы нужно положить в папку: '/content/drive/MyDrive/Motorica/' либо изменить переменную PATH с указанием пути расположения файла, если файлы ноутбуков открываются не через Colab.


