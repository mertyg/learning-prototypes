from utils import convert_to_bags, load_data
import numpy as np

def get_data(folder, dataset, rep, fold):
    train, test = load_data(folder=folder,
                            dataset=dataset,
                            rep=rep,
                            fold=fold)

    bags_train, labels_train = convert_to_bags(train)
    bags_test, labels_test = convert_to_bags(test)
    bags_train = np.array(bags_train)
    labels_train = np.array(labels_train)
    bags_test = np.array(bags_test)
    labels_test = np.array(labels_test)

    return bags_train, labels_train, bags_test, labels_test
