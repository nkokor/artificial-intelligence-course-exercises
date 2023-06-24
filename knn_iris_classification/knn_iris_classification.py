from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#loading data
iris = load_iris()
X = iris.data
Y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

#viewing types of characteristics and labels
print('Characteristics: ', feature_names)
print('Labels: ', target_names)

#viewing first 5 rows of train characteristics data
print('\nFirst 5 train data rows:\n', X[:5])

#splitting data to train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)


#knn classifier defining and training
classifier_knn = KNeighborsClassifier(n_neighbors = 3)
classifier_knn.fit(X_train, Y_train)

#accuracy check
Y_pred = classifier_knn.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))