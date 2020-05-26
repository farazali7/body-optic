import glob
import os
import numpy as np
import cv2

n_classes = 11
n_examples_train = 4
n_examples_test = 1
width, height, channels = 224, 224, 3

root_path = "raw_dataset/Female"
train_path = os.path.join(root_path, 'Train')
test_path = os.path.join(root_path, 'Test')

train_dirs = [f for f in glob.glob(os.path.join(train_path, '*')) if os.path.isdir(f)]
test_dirs = [f for f in glob.glob(os.path.join(test_path, '*')) if os.path.isdir(f)]

assert len(train_dirs) == n_classes
assert len(test_dirs) == n_classes

read_and_resize = lambda x: cv2.resize(cv2.imread(x, 1), (width, height))

def format_dataset(dataset, dirs, name='train'):
    for i, d in enumerate(dirs):
        fs = np.asarray(glob.glob(os.path.join(d, '*.PNG')))
        n_examples = n_examples_train
        if name == 'test':
            n_examples = n_examples_test
        fs = fs[np.random.permutation(len(fs))][:n_examples]
        for j, f in enumerate(fs):
            dataset[i, j] = read_and_resize(f)
        print('{}: {} of {}'.format(name, i + 1, len(dirs)))
    return dataset

train_dataset = np.zeros((n_classes, n_examples_train, width, height, channels), dtype=np.float)
train_dataset = format_dataset(train_dataset, train_dirs)
np.save('datasets/female_dataset_train.npy', train_dataset)
del train_dataset

test_dataset = np.zeros((n_classes, n_examples_test, width, height, channels), dtype=np.float)
test_dataset = format_dataset(test_dataset, test_dirs, name='test')
np.save('datasets/female_dataset_test.npy', test_dataset)
