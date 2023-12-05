import keras
import tensorflow as tf
import sys
import h5py
import numpy as np
from collections import Counter


class G(tf.keras.Model):
    def __init__(self, B, B_prime):
        super(G, self).__init__()
        self.B = B
        self.B_prime = B_prime

    def predict(self, value):
        y = np.argmax(self.B.predict(value), axis=1)
        y_prime = np.argmax(self.B_prime.predict(value), axis=1)
        pred = np.zeros(value.shape[0])
        for i in range(value.shape[0]):
            if y[i] == y_prime[i]:
                pred[i] = y[i]
            else:
                pred[i] = 1283
        return pred


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def prune_badnet(model, validation_data, pruing_threshold):
    origin_predict = model.predict(validation_data[0])
    origin_predict_label = np.argmax(origin_predict, axis=1)
    origin_acc = np.mean(np.equal(origin_predict_label, validation_data[1]))
    threshold = origin_acc - pruing_threshold
    print(threshold)
    ## get average activation get hte output right after the last pooling layer "pool_3"
    temp_model = keras.models.Model(inputs=model.input, outputs=model.get_layer('pool_3').output)
    pool_output = temp_model.predict(validation_data[0])
    avg_activation = tf.math.reduce_mean(pool_output, axis=[0, 1, 2])
    idx_to_prune = tf.argsort(avg_activation)
    last_conv_weight = model.get_layer('conv_3').get_weights()[0]
    last_conv_biases = model.get_layer('conv_3').get_weights()[1]
    prune_count = 0
    for idx in idx_to_prune:
        last_conv_weight[:, :, :, idx] = 0
        last_conv_biases[idx] = 0

        pruned_model = keras.models.clone_model(model)
        pruned_model.set_weights(model.get_weights())
        pruned_model.get_layer('conv_3').set_weights([last_conv_weight, last_conv_biases])

        prune_predict = pruned_model.predict(validation_data[0])
        prune_label = np.argmax(prune_predict, axis=1)
        prune_acc = np.mean(np.equal(prune_label, validation_data[1]))
        print(prune_acc)
        prune_count += 1
        if prune_acc < threshold:
            break
        model = pruned_model
    print(f'prune Count for X= {pruing_threshold} is {prune_count}')
    pruned_model.save(f'./goodnet{pruing_threshold}.h5')

    return model


x_valid_data, y_valid_data = data_loader('./data/cl/valid.h5')
print(len(Counter(y_valid_data).keys()))
print(y_valid_data.shape)
for x in [0.02, 0.04, 0.10]:
    badnet_model = keras.models.load_model('./models/bd_net.h5')
    goodnet = prune_badnet(badnet_model, (x_valid_data, y_valid_data), x)

badnet_model = keras.models.load_model('./models/bd_net.h5')
repaired_goodnet_002 = keras.models.load_model('./goodnet0.02.h5')
repaired_goodnet_004 = keras.models.load_model('./goodnet0.04.h5')
repaired_goodnet_010 = keras.models.load_model('./goodnet0.1.h5')

x_test_cl_data, y_test_cl_data = data_loader('./data/cl/test.h5')
x_test_bd_data, y_test_bd_data = data_loader('./data/bd/bd_test.h5')

G_002 = G(badnet_model, repaired_goodnet_002)
G_004 = G(badnet_model, repaired_goodnet_004)
G_010 = G(badnet_model, repaired_goodnet_010)


clean_label_2 = G_002.predict(x_test_cl_data)
clean_acc_002 = np.mean(np.equal(clean_label_2, y_test_cl_data)) * 100
print(f'G 002 Clean Classification accuracy: {clean_acc_002}')

bd_label_2 = G_002.predict(x_test_bd_data)
bd_acc_002 = np.mean(np.equal(bd_label_2, y_test_bd_data)) * 100
print(f'G 002 Attack Success Rate: {bd_acc_002}')

clean_label_4 = G_004.predict(x_test_cl_data)
clean_acc_004 = np.mean(np.equal(clean_label_4, y_test_cl_data)) * 100
print(f'G 004 Clean Classification accuracy: {clean_acc_004}')

bd_label_4 = G_004.predict(x_test_bd_data)
bd_acc_004 = np.mean(np.equal(bd_label_4, y_test_bd_data)) * 100
print(f'G 004 Attack Success Rate: {bd_acc_004}')

clean_label_1 = G_010.predict(x_test_cl_data)
clean_acc_010 = np.mean(np.equal(clean_label_1, y_test_cl_data)) * 100
print(f'G 010 Clean Classification accuracy: {clean_acc_010}')

bd_label_1 = G_010.predict(x_test_bd_data)
bd_acc_010 = np.mean(np.equal(bd_label_1, y_test_bd_data)) * 100
print(f'G 010 Attack Success Rate: {bd_acc_010}')
