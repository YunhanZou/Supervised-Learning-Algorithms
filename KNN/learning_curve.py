from DataInterface import get_car_dataset, get_pendigits_dataset, split_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
	
	dataset = get_car_dataset()
	data = split_dataset(dataset, 0.25)
	train, test = data
	features_test, labels_test = test
	k = 40

	training_accuracy = []
	testing_accuracy = []

	for train_size in range(6, 99, 2):
		train_size /= 100.0
		data = split_dataset(train, 1 - train_size)
		train2, test2 = data
		features_train, labels_train = train2
		model = KNeighborsClassifier(k, weights='distance')
		model.fit(features_train,labels_train)

		predictions = model.predict(features_train)
		training_accuracy.append(accuracy_score(labels_train, predictions))
		predictions = model.predict(features_test)
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
	X = [i / 100.0 for i in range(6, 99, 2)]
	for x in X:
		print(str(x) + ", "),
