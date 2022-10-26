# Motorica Advanced Gesture Classification
## Продвинутая задача классификации жестов
__________________________________________

Проект реализован в последовательности ноутбуков и набора файлов:

**1)** [*1_EDA_sprint_3.ipynb*](https://github.com/Alex1iv/Motorica_3/blob/main/1_EDA_sprint_3.ipynb) - EDA с анализом тренировочных и тестовых данных до начала работы с моделью, в т.ч. анализ показаний датчиков и жестов.

Используемые в ноутбуке функции приведены в файле [functions.py](https://github.com/Alex1iv/Motorica_3/blob/main/functions.py), который должен находиться в той же папке, что и [*1_EDA_sprint_3.ipynb*](https://github.com/Alex1iv/Motorica_3/blob/main/1_EDA_sprint_3.ipynb).


**Возможно что-то еще нужно дописать после доработки файла**  


**2)** В файле [*2_model_nn.ipynb*](https://github.com/Alex1iv/Motorica_3/blob/main/2_model_nn.ipynb) развит подход к решению задачи на базе **baseline** с применением двух моделей нейросети. 

В первой части ноутбука данные загружаются из архива и преобразуются с помощью библиотеки mne для последующей подачи данных на обучение. Далее последовательно обучаются две модели: 

- SimpleRNN (первая модель на базе слоя SimpleRNN библиотеки Keras); 

- LSTM (вторая модель, в ее основе лежат несколько слоев LSTM библиотеки Keras и дополнительный Dense-слой). 

Важно отметить, что тренировочные и тестовые данные имеют разделение на 3 ряда данных и по каждому набору происходит параллельное обучение группы моделей, имеющих одинаковую структуру и набор параметров. 

Основная задача работы первой модели - определить фактический момент изменения жеста (появление "ступеньки") по данным X_train для последующего обучения более сложной модели. Использование упрощенной модели SimpleRNN совместно с использованием loss="mean_squared_error" и функцией активации 'sigmoid' (activation='sigmoid') в выходном слое при сборке модели позволяет сделать предсказание "ступеньки" при решении задачи классификации жестов по данным датчиков (X_train). Модель учитывает классы из y_train, а время выполнения движения определяется из предикта по X_train как момент изменения класса (жеста). 

Необходимость первого этапа обусловлена спецификой подготовки данных для обучения, когда человек ("пилот") с зафиксированным на запястье набором датчиков повторяет жесты следуя командам манипулятора. Таким образом, изначально y_train представляет собой момент подачи манипулятором команды на изменение жеста, а данные X_train - фактическое выполнение жеста - запаздывают на некоторое время относительно исходного y_train. 

Для того, чтобы компенсировать ошибки предсказания первой модели, обучение SimpleRNN по каждому "пилоту" проводится несколько раз с разными параметрами validation_split и затем результаты предсказания каждой модели усредняются по каждому пилоту. 

Обучение второй модели производится на оригинальных данных X_train и корректированных данных y_train_ch (предсказание обученной модели SimpleRNN на X_train). Далее обученная модель LSTM используется для предсказания тестовых данных. 

При работе с моделями для управления обучением (выбор лучшей модели, изменение learning_rate, остановка обучения при выходе на плато) используется набор функций *callbacks* библиотеки Keras.

В целях обеспечения повторимости результатов и подбора гиперпараметров в начале ноутбука и при каждом сбросе сессии (tf.keras.backend.clear_session(): Resets all state generated by Keras) устанавливается исходное значение seed_value.  

В ноутбуке оставлены закомментированные ячейки с пометками и пояснениями для сохранения возможности запуска в Google Colab.

**3)** [*3_rnn_baseline.ipynb*](https://github.com/Alex1iv/Motorica_3/blob/main/3_rnn_baseline.ipynb)- ноутбук, предоставленный организаторами соревнования в качестве **baseline**.   

**4)** Папка [*data*](https://github.com/Alex1iv/Motorica_3/tree/main/data) содержит архив с исходными данными:

- X_train_1.npy, X_train_2.npy, X_train_3.npy: файлы с тренировочными данными ("фичи", показания датчиков по каждому "пилоту");

- y_train_1.npy, y_train_2.npy, y_train_3.npy: файлы с тренировочными "таргетами" (от манипулятора);

- X_test_dataset_1.pkl, X_test_dataset_2.pkl, X_test_dataset_3.pkl: файлы тестовых данных ("фичи", показания датчиков по каждому "пилоту") для предсказания и сабмита;

- sample_submission.csv: файл примера загрузки предсказанных данных на Kaggle.


