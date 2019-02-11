import math
import random

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


def get_pendigits_dataset(start=None, end=None):

	data = open('datasets/pendigits.txt')

	examples = {}
	examples['d1'] = []
	examples['d2'] = []
	examples['d3'] = []
	examples['d4'] = []
	examples['d5'] = []
	examples['d6'] = []
	examples['d7'] = []
	examples['d8'] = []
	examples['d9'] = []
	examples['d10'] = []
	examples['d11'] = []
	examples['d12'] = []
	examples['d13'] = []
	examples['d14'] = []
	examples['d15'] = []
	examples['d16'] = []

	coding = {}
	coding[0] = 'd1'
	coding[1] = 'd2'
	coding[2] = 'd3'
	coding[3] = 'd4'
	coding[4] = 'd5'
	coding[5] = 'd6'
	coding[6] = 'd7'
	coding[7] = 'd8'
	coding[8] = 'd9'
	coding[9] = 'd10'
	coding[10] = 'd11'
	coding[11] = 'd12'
	coding[12] = 'd13'
	coding[13] = 'd14'
	coding[14] = 'd15'
	coding[15] = 'd16'

	labels = []

	num_attr = 16
	for line in data:
		count = 0
		for val in line.split(','):
			if count < num_attr:
				examples[coding[count]].append(float(val))
				count += 1
			else:
				labels.append(float(val[:-1]))

	return zip(examples['d1'], examples['d2'], examples['d3'], examples['d4'], 
		examples['d5'], examples['d6'], examples['d7'], examples['d8'], examples['d9'], 
		examples['d10'], examples['d11'], examples['d12'], examples['d13'], examples['d14'], 
		examples['d15'],examples['d16']), labels


def get_car_dataset(start=None, end=None):
    """
    Reads in and parses through the dataset.

    Args:
        start (int): optional line number to start dataset at
        end (int): optional line number to end dataset at
    Returns:
        list<dictionary<str,str>>: list of examples as dictionaries
        dictionary<str,list<str>>: a dictionary mapping each attribute to all of its possible values
        str: the name of the label
        list<str>>: the list of possible label values

    """

    data = open("datasets/car.data.txt")

    examples = {}
    examples['buying'] = []
    examples['maint'] = []
    examples['doors'] = []
    examples['persons'] = []
    examples['lug_boot'] = []
    examples['safety'] = []

    coding = {}
    coding[0] = 'buying'
    coding[1] = 'maint'
    coding[2] = 'doors'
    coding[3] = 'persons'
    coding[4] = 'lug_boot'
    coding[5] = 'safety'

    clarification = []

    num_attr = 6
    for line in data:
        count = 0
        for val in line.split(','):
            if count < num_attr:
                examples[coding[count]].append(val)
                count += 1
            else:
                clarification.append(val[:-1])
                break

    le = preprocessing.LabelEncoder()
    buying_encoded = le.fit_transform(examples['buying'])
    maint_encoded = le.fit_transform(examples['maint'])
    doors_encoded = le.fit_transform(examples['doors'])
    persons_encoded = le.fit_transform(examples['persons'])
    lug_boot_encoded = le.fit_transform(examples['lug_boot'])
    safety_encoded = le.fit_transform(examples['safety'])
    class_encoded = le.fit_transform(clarification)

    return zip(buying_encoded, maint_encoded, doors_encoded, persons_encoded, lug_boot_encoded, safety_encoded), class_encoded


def split_dataset(data, ratio):
	
	features, labels = data
	l = len(features)
	test_size = int(math.floor(l * ratio))

	test_indices = random.sample(range(l), test_size)
	train_indices = [ind for ind in range(l) if ind not in test_indices]
	features_train = [features[i] for i in train_indices]
	labels_train = [labels[i] for i in train_indices]
	features_test = [features[i] for i in test_indices]
	labels_test = [labels[i] for i in test_indices]

	return (features_train, labels_train), (features_test, labels_test)


