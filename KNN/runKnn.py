from DataInterface import get_car_dataset, get_pendigits_dataset, split_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time


def train_KNN(data, k):

	train, test = data
	features, labels = train
	model = KNeighborsClassifier(k, weights='distance')
	model.fit(features,labels)

	return model


def test_KNN(data, model):

	train, test = data
	features, labels = test
	predictions = model.predict(features)
	
	return accuracy_score(labels, predictions)


if __name__ == '__main__':

	print("Now training and testing KNN on car dataset:\n")
	dataset = get_car_dataset()
	num_iter = 10
	accuracies = 0
	start_time = time.time()
	k = 40

	print('k is set to ' + str(k) + '\n')
	for _ in range(num_iter):
		data = split_dataset(dataset, 0.25)
		model = train_KNN(data, k)
		accuracy = test_KNN(data, model)
		accuracies += accuracy

	duration = time.time() - start_time
	print("The average classification rate is {0:.3f}.\n".format(accuracies / num_iter))
	print("The run time is " + str(duration) + " sec.")

	print("\nComplete.\n-------------------------\n")

	print("Now training and testing KNN on pen digits dataset:\n")
	dataset = get_pendigits_dataset()
	accuracies = 0
	k = 100

	print('k is set to ' + str(k) + '\n')
	start_time = time.time()
	for _ in range(num_iter):
		data = split_dataset(dataset, 0.25)
		model = train_KNN(data, k)
		accuracy = test_KNN(data, model)
		accuracies += accuracy

	duration = time.time() - start_time
	print("The average classification rate is {0:.3f}.\n".format(accuracies / num_iter))
	print("The run time is " + str(duration) + " sec.")

	print("Complete.\n")
