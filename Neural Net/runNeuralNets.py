from DataInterface import get_pendigits_dataset, get_car_dataset, split_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time


def training(train_data, hidden_layer_sizes):

	features_train, labels_train = train_data

	mlp = MLPClassifier(hidden_layer_sizes, max_iter=200)
	mlp.fit(features_train, labels_train)

	return mlp


def testing(mlp, test_data):

	features_test, labels_test = test_data
	predictions = mlp.predict(features_test)

	return accuracy_score(labels_test, predictions)


if __name__ == '__main__':

	num_iter = 10  # number of test/train splits

	print("\nNow training and testing on the car dataset with " + str(num_iter) + " runs of train/test splits:\n")
	data = get_car_dataset()
	hidden_layers = (10,10,10,10)
	print("The neural network has {} hidden layers, each layer has size: ".format(len(hidden_layers))),
	for layer in hidden_layers:
		print(layer),
	print('\n')
	accuracies = 0
	start_time = time.time()

	for _ in range(num_iter):
		train, test = split_dataset(data, 0.25)
		mlp = training(train, hidden_layers)
		accuracy = testing(mlp, test)
		accuracies += accuracy

	duration = time.time() - start_time
	print("The average classification rate is {0:.3f}.\n".format(accuracies / num_iter))
	print("The run time is " + str(duration) + " sec.")
	print("\nComplete\n----------------------------------\n")

	print("Now training and testing on the pen digits dataset " + str(num_iter) + " runs of train/test splits:\n")
	data = get_pendigits_dataset()
	hidden_layers = (10,10)
	print("The neural network has {} hidden layers, each layer has size: ".format(len(hidden_layers))),
	for layer in hidden_layers:
		print(layer),
	print('\n')
	accuracies = 0
	start_time = time.time()

	for i in range(num_iter):
		train, test = split_dataset(data, 0.25)
		mlp = training(train, hidden_layers)
		accuracy = testing(mlp, test)
		accuracies += accuracy

	duration = time.time() - start_time
	print("The average classification rate is {0:.3f}.\n".format(accuracies / num_iter))
	print("The run time is " + str(duration) + " sec.")
	print("Complete\n\n----------------------------------\n")

