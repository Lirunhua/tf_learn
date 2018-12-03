import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

for one_element in tfe.Iterator(dataset):
    print(one_element)