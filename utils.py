import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def convert_to_bags(data,
                    split_instances=False,
                    instance_norm=True,
                    split_ratio=0.2,
                    stride_ratio=0.5):
  bags = []
  labels = []
  current_bag = []
  current_label = data[0, 0]
  cur = data[0, 1]
  instance_size = np.round(split_ratio * data[0, 2:].shape[0]).astype("int")
  stride = np.round(stride_ratio * instance_size).astype("int")

  for i in range(data.shape[0]):
    if data[i, 1] == cur:
      instance = data[i, 2:]
      if instance_norm:
        instance = (instance - np.mean(instance)) / (1e-08 + np.std(instance))
      if split_instances:
        size = instance.shape[0]
        window = instance_size
        while True:
          current_bag.append(instance[window - instance_size:window])
          window += stride
          if window >= size:
            window = size
            current_bag.append(instance[window - instance_size:window])
            break
      else:
        current_bag.append(instance)
    else:
      bags.append(np.array(current_bag))
      labels.append(np.array(current_label))
      current_label = data[i, 0]
      current_bag = []
      instance = data[i, 2:]
      if instance_norm:
        instance = (instance - np.mean(instance)) / (1e-08 + np.std(instance))
      if split_instances:
        size = instance.shape[0]
        window = instance_size
        while True:
          current_bag.append(instance[window - instance_size:window])
          window += stride
          if window >= size:
            window = size
            current_bag.append(instance[window - instance_size:window])
            break
      else:
        current_bag.append(instance)
      cur = data[i, 1]
  bags.append(np.array(current_bag))
  labels.append(np.array(current_label, dtype="int32"))
  return bags, labels


def load_data(folder, dataset, rep, fold):
  file = "".join([dataset, ".csv"])
  filepath = os.path.join(folder, file)
  data = np.genfromtxt(filepath, delimiter=",")

  #Normalizing data
  cv_file = "".join([dataset, ".csv_rep", str(rep), "_fold", str(fold), ".txt"])
  cv_path = os.path.join(folder, cv_file)
  testind = np.loadtxt(cv_path)
  testind = testind.astype(int)

  mask = np.isin(data[:, 1], testind)
  testcv = data[mask, :]
  traincv = data[np.logical_not(mask), :]
  return traincv, testcv


def plot_prototypes(prots, savefile=None):

  n_prots = prots.shape[0]
  fig, axs = plt.subplots(ceil(n_prots / 3),
                          3,
                          figsize=(15, 6),
                          facecolor='w',
                          edgecolor='k')
  fig.subplots_adjust(hspace=.5, wspace=.001)
  fig.suptitle("Dataset Prototypes")
  axs = axs.ravel()
  for i in range(n_prots):
    axs[i].plot(prots[i])
  if savefile:
    dataset = savefile.split("_")[1]
    fig.suptitle(dataset + " Dataset Prototypes")
    plt.savefig(savefile)
  else:
    plt.show()
