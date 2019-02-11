from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree
from DataInterface import get_car_dataset, get_pendigits_dataset, split_dataset

from sklearn.ensemble import AdaBoostClassifier
import time


def train_boosting(data, max_depth):

	train, test = data
	features_train, labels_train = train

	dt = DecisionTreeClassifier(criterion="gini", max_depth=max_depth)
	clf = AdaBoostClassifier(n_estimators=50, base_estimator=dt, learning_rate=1)

	clf.fit(features_train, labels_train)

	return clf


def test_boosting(data, clf):

	train, test = data
	features_test, labels_test = test

	predictions = clf.predict(features_test)

	return accuracy_score(labels_test, predictions)


def save_trees_as_png(clf, feature_names, pic_id, num_trees=2):

	trees = clf.estimators_
	count = 0

	for tree in trees:
		dot_data = StringIO()
		filename = "dt_" + pic_id + str(count) + ".png"

		export_graphviz(tree, out_file=dot_data,
				feature_names=feature_names,
                filled=True, rounded=True,
                special_characters=True)

		graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
		graph.write_png(filename)
		count += 1

		if count == num_trees:
			break


if __name__ == '__main__':

	print("\nNow training and testing boosted version of decision tree on car dataset with train/test split for 10 times:\n")

	dataset = get_car_dataset()
	feature_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
	max_depth = 4
	print("To prune the decision tree, the maximum depth is set to " + str(max_depth))

	start_time = time.time()
	accuracies = 0
	for _ in range(10):
		data = split_dataset(dataset, 0.3)
		train, test = data
		features_train, labels_train = train
		clf = train_boosting(data, max_depth)
		accuracy = test_boosting(data, clf)
		accuracies += accuracy

		# save_trees_as_png(clf, feature_names, 'car')
		# print("Trees visualization written to the current folder.")
	duration = time.time() - start_time
	print("The average training accuracy over 10 runs is {0:.3f}.\n".format(accuracies / 10))
	print("The run time is " + str(duration) + " sec.")
	print("\nComplete.\n")
	print("-----------------------------------------------------------------\n")

	print("Now training and testing boosted version of decision tree on pen digits data set with train/test split for 10 times:\n")

	dataset = get_pendigits_dataset()
	feature_names = ["X1", "Y1", "X2", "Y2", "X3", "Y3", 
					"X4", "Y4", "X5", "Y5", "X6", "Y6", 
					"X7", "Y7", "X8", "Y8"]
	max_depth = 4
	print("To prune the decision tree, the maximum depth is set to " + str(max_depth))
	accuracies = 0
	start_time = time.time()

	for _ in range(10):
		data = split_dataset(dataset, 0.3)
		train, test = data
		features_train, labels_train = train
		clf = train_boosting(data, max_depth)
		accuracy = test_boosting(data, clf)
		accuracies += accuracy

	duration = time.time() - start_time
	print("The average training accuracy over 10 runs is {0:.3f}.\n".format(accuracies / 10))
	print("The run time is " + str(duration) + " sec.")
	print("\nComplete.\n")
	print("-----------------------------------------------------------------\n")

	# save_trees_as_png(clf, feature_names, "pendigits")
	# print("Trees visualization written to the current folder.")

	print("\nCopmlete.\n")

