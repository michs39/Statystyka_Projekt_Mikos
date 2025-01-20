import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import datasets

NUMBER_OF_TYPES = 3
TYPE_COLUMN_NAME = "target"


class NaiveBayesClassifier:
    general_box = {}
    flower_boxes = [{} for i in range(NUMBER_OF_TYPES)]
    general_count = 0
    counts = [0 for i in range(NUMBER_OF_TYPES)]
    cols = []

    def fit(self, train_data):
        grouped = train_data.groupby(TYPE_COLUMN_NAME)
        flower_type_dfs = []
        for i in range (NUMBER_OF_TYPES):
            flower_type_dfs.append(grouped.get_group(i))
        for df in flower_type_dfs:
            df.drop([TYPE_COLUMN_NAME],axis=1,inplace = True)
        train_data.drop([TYPE_COLUMN_NAME],axis=1,inplace = True)
        for i in range(NUMBER_OF_TYPES):
            self.counts[i] = len(flower_type_dfs[i])
        self.general_count = len(train_data)
        
        for kol in train_data.keys():
            self.cols.append(kol)

        for kol in self.cols:
            self.general_box[kol] = {}
            for box in self.flower_boxes:
                box[kol] = {}
         
        for kol in self.cols:
            for i in range(NUMBER_OF_TYPES):
                mean = flower_type_dfs[i][kol].mean()
                std = flower_type_dfs[i][kol].std()
                self.flower_boxes[i][kol]['mean'] = mean
                self.flower_boxes[i][kol]['std'] = std

    def predict_proba(self, iris):
        type_probability = [0 for i in range(NUMBER_OF_TYPES)]
        for i in range(NUMBER_OF_TYPES):
            type_probability[i] = self.counts[i] / self.general_count
            for kol in self.cols:
                mean = self.flower_boxes[i][kol]['mean']
                std = self.flower_boxes[i][kol]['std']
                value = iris[kol]
                prob = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((value-mean)**2 / (2 * std**2 )))
                type_probability[i] = type_probability[i] * prob
                #W razie problemów z obliczeniami można zamienić mnożenie na dodawanie logarytmów
        return type_probability
    
    def predict(self,iris):
        type_probability = self.predict_proba(iris)
        most_probable = 0
        for i in range(NUMBER_OF_TYPES):
            if(type_probability[i]>type_probability[most_probable]):
                most_probable = i
        return most_probable

#Main
iris = datasets.load_iris(as_frame=True)["frame"]

#iris.drop(["petal length (cm)"],axis=1,inplace = True)
#iris.drop(["petal width (cm)"],axis=1,inplace = True)
#Planowałem nie brać pod uwagę tych kolumn ponieważ są zależne od siebie i pozostałych kolumn,
#a założeniem naiwnego klasyfikatora bayesowskiego jest niezależność,
#natomiast po porównaniu wyników doszedłem do wniosku, że zostawienie będzie lepszym rozwiązaniem
#ponieważ generuje lepsze wyniki

print(iris)
train_iris, test_iris = train_test_split(iris,train_size=0.7)
test_iris = shuffle(test_iris)
test_iris.reset_index(drop=True,inplace=True)

first_try = NaiveBayesClassifier()
first_try.fit(train_iris)

mistakes = 0
for i in range (len(test_iris)):
    real_type = test_iris[TYPE_COLUMN_NAME][i]
    predict_type = first_try.predict(test_iris.iloc[i])
    probabilities = first_try.predict_proba(test_iris.iloc[i])
    if(real_type != predict_type):
        mistakes = mistakes + 1
    print(real_type, predict_type, probabilities)

print("Accuracy: " , round(100 * (len(test_iris) - mistakes) / len(test_iris),2) , "%")
# Program przewiduje gatunek irysa z dokładnością 90% - 97%,