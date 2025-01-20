import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:
    general_box = {}
    edible_box = {}
    poisonous_box = {}
    poisonous_count = 0
    edible_count = 0
    cols = None

    def fit(self, train_data, possible_values):
        grouped = train_data.groupby("class")
        df_edible = grouped.get_group("e")
        df_poisonous = grouped.get_group("p")
        df_edible.drop(["class"],axis=1,inplace = True)
        df_poisonous.drop(["class"],axis=1,inplace = True)
        train_data.drop(["class"],axis=1,inplace = True)
        self.poisonous_count = len(df_poisonous)
        self.edible_count = len(df_edible)
        
        self.cols = []
        for kol in train_data.keys():
            self.cols.append(kol)
            self.edible_box[kol] = {}
            self.poisonous_box[kol] = {}
            self.general_box[kol] = {}
            
         
        for kol in self.cols:
            for typ in possible_values[kol]:
                edible_count_typ = (df_edible[kol] == typ).sum()
                self.edible_box[kol][typ] = (edible_count_typ ) / self.edible_count
                poisonous_count_typ = (df_poisonous[kol] == typ).sum()
                self.poisonous_box[kol][typ] = (poisonous_count_typ ) / self.poisonous_count
                count_typ = (train_data[kol] == typ).sum()
                self.general_box[kol][typ] = (count_typ ) / (self.poisonous_count + self.edible_count)
                #Można dodawać 1 dla każdej ilości aby nie zepsuć wyników mnożeniem przez 0,
                #ale w tym przypadku bez tego wyniki są lepsze



    def predict_proba(self, mushroom):
        edible_probability = self.edible_count/(self.edible_count + self.poisonous_count)
        poisonous_probability = self.poisonous_count/(self.edible_count + self.poisonous_count)
        for kol in self.cols:
            edible_probability = edible_probability * self.edible_box[kol][mushroom[kol]] / self.general_box[kol][mushroom[kol]]
            poisonous_probability = poisonous_probability * self.poisonous_box[kol][mushroom[kol]] / self.general_box[kol][mushroom[kol]]
        return (round(edible_probability.item(),6) , round(poisonous_probability.item(),6))
    
    def predict(self,mushroom):
        result = self.predict_proba(mushroom)
        if(result[0] > result[1]):
            return('e')
        elif(result[1] > result[0]):
            return('p')
        else:
            return None

#Main
mushrooms = pd.read_csv('mushrooms.csv')
mushrooms.drop(["veil-type"],axis=1,inplace = True)
train_mushrooms, test_mushrooms = train_test_split(mushrooms,train_size=0.5)
test_mushrooms = shuffle(test_mushrooms)
test_mushrooms.reset_index(drop=True,inplace=True)

possible_values = {}
for key in mushrooms.keys():
    possible_values[key] = []
    unique_values = mushrooms[key].unique()
    for val in unique_values:
        possible_values[key].append(val)

first_try = NaiveBayesClassifier()
first_try.fit(train_mushrooms,possible_values)

mistakes = 0
for i in range (len(test_mushrooms)):
    real_type = test_mushrooms['class'][i]
    predict_type = first_try.predict(test_mushrooms.iloc[i])
    probabilities = first_try.predict_proba(test_mushrooms.iloc[i])
    if(real_type != predict_type):
        mistakes = mistakes + 1
    print(real_type, predict_type, probabilities)

print("Accuracy: " , round(100 * (len(test_mushrooms) - mistakes) / len(test_mushrooms),2) , "%")

# Program przewiduje jadalność grzybów z dokładnością bliską 100%,





