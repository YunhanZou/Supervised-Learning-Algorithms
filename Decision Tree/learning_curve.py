from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree
# from sklearn.tree import _tree
# from sklearn.externals.six import StringIO 
# from sklearn.tree import export_graphviz
from DataInterface import get_car_dataset, get_pendigits_dataset, split_dataset


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
		dt = DecisionTreeClassifier(criterion="gini", max_depth=10)
		dt.fit(features_train, labels_train)

		predictions = dt.predict(features_train)
		training_accuracy.append(accuracy_score(labels_train, predictions))
		predictions = dt.predict(features_test)
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
