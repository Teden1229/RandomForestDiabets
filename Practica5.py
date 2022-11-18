#ЭТОТ КОД СОЗДАН ДЛЯ ПРОВЕРКИ РБОТЫ В РЕПОЗИТОРИЯХ
#НУЖНО УЗНАТЬ КАК РАБОТАЮТ КОММИТЫ
#ПОСМОТРЕТЬ ИСТОРИЮ ПРОЕКТА
import sklearn
import numpy as np
import pandas as pd
from sklearn import preprocessing
#Предобработка данных
data = (pd.read_csv('diabetes_data_upload.csv')).values #Загрузка данных и преобразование в numpy массив

Y = data[:,16] #Выходные значения class
X = data[:,0:16] #Входные значения

enc = preprocessing.OneHotEncoder()
enc.fit(X[:,1:16]) 
X1 = np.column_stack((X[:,0], enc.transform(X[:,1:16]).toarray())) 
X1[:,0] = preprocessing.scale(X1[:,0]) #Стандартизация значений возраста

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y1 = labelencoder.fit_transform(Y)

#Подготовка тестовой и обучающей выборок
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.2, random_state= 9) #Шафл - перемешивание данных, за это отвечает random_state-начальное значение внутреннему генератору случайных чисел функции 

#Обучение модели
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0).fit(X_train, Y_train) #fit - Подгонка модели в соответствии с заданными обучающими данными.
Y_pred = model.predict(X_test) #Получаем предсказания
print(Y_pred)
Y_pred_proba = model.predict_proba(X_test)

#Оценка точности модели
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred)) #Точность
from sklearn.metrics import recall_score
print(recall_score(Y_test, Y_pred)) #Полнота
from sklearn.metrics import f1_score
print(f1_score(Y_test, Y_pred, average=None)) #F-мера
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred)) #Матрица неточностей
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(model, X_test, Y_test) #График матрицы неточностей  
plt.show()

from sklearn.metrics import roc_curve #чувствительность модели к разным порогам классификации
Y_pred_1 = model.predict_proba(X_test)[:, 1] #вероятности только для положительного исхода
fpr, tpr, treshold = roc_curve(Y_test, Y_pred_1) #Расчет roc-кривой
plt.plot(fpr, tpr, color='darkorange') 
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.show()

#Улучшение модели
model1 = LogisticRegression(random_state=0, C=100).fit(X_train, Y_train)
Y_pred1 = model1.predict(X_test) #Получаем предсказания
print(accuracy_score(Y_test, Y_pred1))
