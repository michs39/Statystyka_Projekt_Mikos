import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns



mushrooms = pd.read_csv('mushrooms.csv')
mushrooms
iris = datasets.load_iris(as_frame=True)["frame"]
grouped = iris.groupby("target")
a = grouped.get_group(0)
b = grouped.get_group(1)
c = grouped.get_group(2)
print (len(b))
print (len(a))
print (len(c))

#for key in mushrooms.keys():
#    mushrooms[key]=mushrooms[key].astype('category').cat.codes
#mushrooms.drop(["veil-type"],axis=1,inplace = True)
#corr = mushrooms.corr()
#plt.figure(figsize=(10,8), dpi =500)
#sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.5)
#print (mushrooms["veil-type"])
#plt.show()

corr = iris.corr()
sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.5)
plt.show()