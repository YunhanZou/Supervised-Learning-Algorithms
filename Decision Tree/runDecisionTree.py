from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from DataInterface import get_car_dataset, get_pendigits_dataset, split_dataset
import time


def train_decision_tree(data, max_depth=10000):

	train, test = data
	features_train, labels_train = train

	clf_gini = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
	clf_gini.fit(features_train, labels_train)

	return clf_gini


def test_decision_tree(data, dt):

	train, test = data
	features_test, labels_test = test

	predictions = dt.predict(features_test)
	
	return accuracy_score(labels_test, predictions)


def count_nodes_without_pruning(data):

	train, test = data
	features_train, labels_train = train

	clf_gini = DecisionTreeClassifier(criterion="gini")
	clf_gini.fit(features_train, labels_train)

	return clf_gini.tree_.node_count


def calc_training_testing_accu(data, dt):

	(features_train, labels_train), (features_test, labels_test) = data

	predictions = dt.predict(features_train)
	print("Training accuracy is {0:.3f}.\n".format(accuracy_score(labels_train, predictions)))
	predictions = dt.predict(features_test)
	print("Testing accuracy is {0:.3f}.\n".format(accuracy_score(labels_test, predictions)))


def save_dt_as_png(dt, feature_names, pic_id):

	dot_data = StringIO()
	filename = "dt_" + pic_id + ".png"

	export_graphviz(dt, out_file=dot_data,
				feature_names=feature_names,
                filled=True, rounded=True,
                special_characters=True)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
	graph.write_png(filename)
	print("Saving tree visualization as png file...")
	print("Tree visualization saved to the project folder. Name: " + filename)


if __name__ == '__main__':
	num_iter = 10

	print("Now training and testing decision tree on car dataset for 10 runs of train/test split:\n")
	features_name = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
	dataset = get_car_dataset()
	accuracies = 0
	max_depth = 10
	start_time = time.time()

	print('To prune the tree, the maximum depth is set to ' + str(max_depth))
	for _ in range(num_iter):
		data = split_dataset(dataset, 0.25)
		train, test = data
		features_train, labels_train = train
		dt = train_decision_tree(data, max_depth)
		accuracy = test_decision_tree(data, dt)
		accuracies += accuracy

	duration = time.time() - start_time
	print("Average accuracy is {0:.3f}.\n".format(accuracies / num_iter))
	print("The run time is " + str(duration) + " sec.")

	print("\nComplete.\n")
	print("-----------------------------------------------------------------\n")
	print("Now training and testing decision tree on pen digits dataset for 10 runs of train/test split:\n")
	feature_names = ["X1", "Y1", "X2", "Y2", "X3", "Y3", 
					"X4", "Y4", "X5", "Y5", "X6", "Y6", 
					"X7", "Y7", "X8", "Y8"]
	dataset = get_pendigits_dataset()
	data = split_dataset(dataset, 0.25)
	accuracies = 0
	max_depth = 10
	print('To prune the tree, the maximum depth is set to ' + str(max_depth))
	start_time = time.time()

	for _ in range(num_iter):
		data = split_dataset(dataset, 0.25)
		train, test = data
		features_train, labels_train = train
		dt = train_decision_tree(data, max_depth)
		accuracy = test_decision_tree(data, dt)
		accuracies += accuracy

	duration = time.time() - start_time
	print("Average accuracy is {0:.3f}.\n".format(accuracies / num_iter))
	print("The run time is " + str(duration) + " sec.")

	print("\nCopmlete.\n")
