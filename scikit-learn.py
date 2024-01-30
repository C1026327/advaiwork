import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

url = "resources/iris.csv"
columnNames = 'id', 'SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species'

dataset = pd.read_csv(url, names=columnNames)

print(dataset.shape)

feature_columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
x = dataset[feature_columns].values
y = dataset['Species'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

