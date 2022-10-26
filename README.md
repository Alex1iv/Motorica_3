# Motorica_3
Проект реализован в последовательности ноутбуков и набора файлов:

    Файл *1_EDA_sprint_3.ipynb* представляет собой EDA с анализом тренировочных и тестовых данных до начала работы с моделью, в т.ч. анализ показаний датчиков и жестов. Используемые в ноутбуке функции приведены в файле functions.py, который должен находиться в той же папке, что и *1_EDA_sprint_3.ipynb*.

    **Возможно что-то еще нужно дописать после доработки файла**  

    Файл *2_model_nn.ipynb* является доработкой baseline, включающий в себя решение задачи с помощью двух моделей нейросети. В первой части ноутбука данные загружаются из архива и преобразуются с помощью библиотеки mne для последующей подачи данных в модель. Далее последовательно обучаются две модели: SimpleRNN (первая модель, в основе лежит слой SimpleRNN библиотеки Keras) и LSTM (вторая модель, в основе лежит несколько слоев LSTM библиотеки Keras и дополнительного Dense-слоя). Важно отметить, что в тренировочные и тестовые данные имеют разделение на 3 набора данных и по каждому набору обучаются отдельные группы моделей.   
    Основная задача работы первой модели - это предсказание изменения жеста (появления  "ступеньки") по данным X_train для последующего обучения более сложной модели. Использование упрощенной модели SimpleRNN совместно с функцией активации 'sigmoid' (activation='sigmoid') в выходном слое и использование loss="mean_squared_error" при сборке модели позволяет сделать предсказание "ступеньки" по данным датчиков (X_train). Модель учитывает классы из y_train, а время начала ступеньки берет из X_train. Примечание: Изначально y_train представляет собой данные по жестам, которые определены манипулятором. При подготовке данных для обучения человек ("пилот") с набором датчиков повторяет жесты за манипулятором, в результате чего данные X_train запаздывают на некоторое время относительно исходного y_train. Для того, чтобы компенсировать ошибки предсказания первой модели, обучение SimpleRNN по каждому "пилоту" проводится несколько раз с разными параметрами validation_split и затем результаты предсказания каждой модели усредняются по каждому пилоту.  
    Вторая модель обучается на X_train и измененных данных y_train_ch (предсказание обученной модели SimpleRNN на X_train). Далее обученная модель LSTM используется для предсказания тестовых данных.  
    При обучении модели для возможности управления обучением (выбор лучшей модели, изменение learning_rate, остановка обучения при выходе на плато) используется набор функций *callbacks* библиотеки Keras.
    Ноутбук можно запускать в Google Colab. Для этого в ноутбуке оставлены закомментированные ячейки с соответствующиеми пометками и пояснениями.  
    Для возможности повторения результатов и подбора гиперпараметров в начале ноутбука и при каждом сбросе сессии (tf.keras.backend.clear_session(): Resets all state generated by Keras) устанавливается исходное значение seed_value.  
    


    Файлы X_train.npy, X_test.npy, y_train.csv, sample_submission.csv - исходные файлы тренировочного, тестового фичей, тренировочного таргета и файла примера загрузки на Kaggle предсказанных данных. Если файлы ноутбука открываются через Colab, то вышеуказанные файлы нужно положить в папку: '/content/drive/MyDrive/Motorica/' либо изменить переменную PATH с указанием пути расположения файла, если файлы ноутбуков открываются не через Colab.


