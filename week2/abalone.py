import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data" )


columns = [
     "Sex",
     "Length",
     "Diameter",
     "Height",
     "Whole weight",
     "Shucked weight",
     "Viscera weight",
     "Shell weight",
     "Rings"
 ]


abaloneAll = pd.read_csv(url, names=columns)

abalone=abaloneAll.drop("Sex", axis=1)
abalone8_12 = abalone.drop(abalone[(abalone.Rings != 8) & (abalone.Rings != 11)].index)
print(abalone8_12["Rings"].value_counts())

x = abalone8_12.iloc[:, :-1].values
y = abalone8_12.iloc[:, 7].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
scaler = StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform (x_train)
x_test = scaler.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))