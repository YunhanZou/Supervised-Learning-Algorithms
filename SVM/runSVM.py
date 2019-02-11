from DataInterface import get_pendigits_dataset, get_car_dataset, split_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time


def train_svm_linear_kernel(data):
	train, test = data
	features_train, labels_train = train
	linear_kernel_svm_clf = Pipeline((
		("svm_clf", SVC(kernel="linear")),
	))

	linear_kernel_svm_clf.fit(features_train, labels_train)

	return linear_kernel_svm_clf


def train_svm_guassian_kernel(data, C=10):

	train, test = data
	features_train, labels_train = train
	rbf_kernel_svm_clf = SVC(kernel='rbf', gamma='auto', C=C)
	rbf_kernel_svm_clf.fit(features_train, labels_train)

	return rbf_kernel_svm_clf


def test_svm(data, svm_clf):

	train, test = data
	features_test, labels_test = test

	predictions = svm_clf.predict(features_test)

	return accuracy_score(labels_test, predictions)


if __name__ == '__main__':

	print("Training and testing SVM on the car dataset:\n---------------------------")
	data = get_car_dataset()
	data = split_dataset(data, 0.25)
	num_iter = 10

	print("1). Using linear kernel:\n")
	accuracies = 0
	start_time = time.time()
	for _ in range(num_iter):
	 	svm_clf_linear_kernel = train_svm_linear_kernel(data)
		accuracy = test_svm(data, svm_clf_linear_kernel)
		accuracies += accuracy
	duration = time.time() - start_time
	print("The average classification rate is {0:.3f}.\n".format(accuracies / num_iter))
	print("The duration is " + str(duration) + " sec.\n")

	print("2). Using Gaussian RBF Kernel:\n")
	accuracies = 0
	start_time = time.time()
	for _ in range(num_iter):
		svm_clf_rbf_kernel = train_svm_guassian_kernel(data, 10)
		accuracy = test_svm(data, svm_clf_rbf_kernel)
		accuracies += accuracy
	duration = time.time() - start_time
	print("The average classification rate is {0:.3f}.\n".format(accuracies / num_iter))
	print("The duration is " + str(duration) + " sec.")

	print("\n Complete.\n-------------------------\n")

	print("Training and testing SVM on the pen digits dataset:\n---------------------------")
	data = get_pendigits_dataset()
	data = split_dataset(data, 0.25)
	print("1). Using linear kernel:\n")
	accuracies = 0
	start_time = time.time()
	for _ in range(1, num_iter):
	 	svm_clf_linear_kernel = train_svm_linear_kernel(data)
		accuracy = test_svm(data, svm_clf_linear_kernel)
		accuracies += accuracy
	duration = time.time() - start_time
	print("The average classification rate is {0:.3f}.\n".format(accuracies / num_iter))
	print("The duration is " + str(duration) + " sec.\n")

	print("2). Using Gaussian RBF Kernel:\n")
	accuracies = 0
	start_time = time.time()
	for _ in range(num_iter):
		svm_clf_rbf_kernel = train_svm_guassian_kernel(data, 5)
		accuracy = test_svm(data, svm_clf_rbf_kernel)
		accuracies += accuracy
	duration = time.time() - start_time
	print("The average classification rate is {0:.3f}.\n".format(accuracies / num_iter))
	print("The duration is " + str(duration) + " sec.")

	print("\nComplete.\n")
