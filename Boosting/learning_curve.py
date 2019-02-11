from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree
from DataInterface import get_car_dataset, get_pendigits_dataset, split_dataset

from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	dataset = get_car_dataset()
	data = split_dataset(dataset, 0.25)
	train, test = data
	features_test, labels_test = test

	training_accuracy = []
	testing_accuracy = []

	for train_size in range(6, 101, 2):
		train_size /= 100.0
		data = split_dataset(train, 1 - train_size)
		train2, test2 = data
		features_train, labels_train = train2
		dt = DecisionTreeClassifier(criterion="gini", max_depth=4)
		clf = AdaBoostClassifier(n_estimators=50, base_estimator=dt, learning_rate=1)
		clf.fit(features_train, labels_train)

		predictions = clf.predict(features_train)
		training_accuracy.append(accuracy_score(labels_train, predictions))
		predictions = clf.predict(features_test)
		testing_accuracy.append(accuracy_score(labels_test, predictions))

	print("Training accuracy: "),
	for num in training_accuracy:
		print(str(num) + ", "),
	print("\n-------------------------------------\n")
	print("Testing accuracy: "),
	for num in testing_accuracy:
		print(str(num) + ", "),
	print("\n-------------------------------------\n")
	print("X: "),
	X = [i / 100.0 for i in range(6, 101, 2)]
	for x in X:
		print(str(x) + ", "),
