from DataInterface import get_car_dataset, split_dataset, get_pendigits_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
	
	dataset = get_pendigits_dataset()
	data = split_dataset(dataset, 0.25)
	train, test = data
	features_test, labels_test = test
	hidden_layers = (10, 10)
	training_accuracy = []
	testing_accuracy = []

	for train_size in range(6, 101, 2):
		if train_size == 100:
			break
		train_size /= 100.0
		data = split_dataset(train, 1 - train_size)
		train2, test2 = data
		features_train, labels_train = train2
		mlp = MLPClassifier(hidden_layers, max_iter=500)
		mlp.fit(features_train, labels_train)

		predictions = mlp.predict(features_train)
		training_accuracy.append(accuracy_score(labels_train, predictions))
		predictions = mlp.predict(features_test)
		testing_accuracy.append(accuracy_score(labels_test, predictions))

	features_train, labels_train = train
	mlp = MLPClassifier(hidden_layers, max_iter=500)
	mlp.fit(features_train, labels_train)
	predictions = mlp.predict(features_train)
	training_accuracy.append(accuracy_score(labels_train, predictions))
	predictions = mlp.predict(features_test)
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
