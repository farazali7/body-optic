"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import numpy as np
import tensorflow as tf
from machine_learning_approaches.prototypical_net_approach.base_prototypical import Prototypical

n_epochs = 5
n_episodes = 5
n_way = 11
n_shot = 3
n_query = 1
n_examples = 4
im_height, im_width, channels = 160, 160, 3
h_dim = 64
z_dim = 64

# Load train dataset
train_dataset = np.load('../../datasets/male_dataset_train.npy')
n_classes = train_dataset.shape[0]

print(train_dataset.shape)

model = Prototypical(n_shot, n_query, im_width, im_height, channels)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.metrics.Mean(name='train_loss')
train_acc = tf.metrics.Mean(name='train_accuracy')

@tf.function
def model_train(support, query):
    with tf.GradientTape() as tape:
        loss, acc = model.call(support, query)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(acc)

    return loss, acc



for ep in range(n_epochs):
    for epi in range(n_episodes):
        epi_classes = np.random.permutation(n_classes)[:n_way]
        support = np.zeros([n_way, n_shot, im_height, im_width, channels], dtype=np.float32)
        query = np.zeros([n_way, n_query, im_height, im_width, channels], dtype=np.float32)
        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_shot + n_query]
            # print('selected = {}'.format(selected))
            # print('i = ' + str(i))
            # print('epi_cls = {}'.format(epi_cls))
            # print('n_shot = ' + str(n_shot))
            # print(selected[:n_shot])
            # selected=[0,0,0,0]
            support[i] = train_dataset[epi_cls, selected[:n_shot]]
            query[i] = train_dataset[epi_cls, selected[n_shot:]]
        # support = np.expand_dims(support, axis=-1)
        # query = np.expand_dims(query, axis=-1)
        print('[epoch {}/{}, episode {}/{}]'.format(ep+1, n_epochs, epi+1, n_episodes), end='')
        loss, acc = model_train(support, query)
        print(' => loss: {:.5f}, acc: {:.5f}'.format(loss, acc))

# Load test dataset
test_dataset = np.load('../../datasets/male_dataset_test.npy')
n_classes = test_dataset.shape[0]

print(test_dataset.shape)

test_loss = tf.metrics.Mean(name='test_loss')
test_acc = tf.metrics.Mean(name='test_accuracy')

n_test_episodes = 20
n_test_way = 11
n_test_shot = 4
n_test_query = 1

print('Testing...')
for epi in range(n_test_episodes):
    epi_classes = np.random.permutation(n_classes)[:n_test_way]
    support = np.zeros([n_test_way, n_test_shot, im_height, im_width, channels], dtype=np.float32)
    query = np.zeros([n_test_way, n_test_query, im_height, im_width, channels], dtype=np.float32)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(n_examples)[:n_test_shot + n_test_query]
        # print('selected = {}'.format(selected))
        # print('i = ' + str(i))
        # print('epi_cls = {}'.format(epi_cls))
        # print('n_shot = ' + str(n_shot))
        # print(selected[:n_shot])
        support[i] = train_dataset[epi_cls, selected[:n_test_shot]]
        query[i] = test_dataset[epi_cls, 0]
    loss, acc = model.call(support, query)
    print(' => loss: {:.5f}, acc: {:.5f}'.format(loss, acc))
    test_loss(loss)
    test_acc(acc)

print('Final Loss: ', test_loss.result().numpy())
print('Final Acc: ', test_acc.result().numpy())

model.save('Proto-model-v1.h5')
