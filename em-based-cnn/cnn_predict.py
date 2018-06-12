'''
	Python 3.5
'''

#some basic imports and setups
import os
import cv2
import time
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from alexnet import AlexNet
# from caffe_classes import class_names

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
	Model Arch Def
'''
#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, 2, None)

#define activation of last layer as score
score = model.fc8

#create op to calculate softmax 
softmax = tf.nn.softmax(score)

'''
	to-do: data IO imgs
'''


'''
	Predict
'''
# to-do
checkpointfile=''

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
saver.restore(sess, checkpointfile)

# Initialize all variables
# sess.run(tf.global_variables_initializer())

for i, image in enumerate(imgs):
    
    img = cv2.imread(image)
    # Convert image to float32 and resize to (227x227)
    img = cv2.resize(img.astype(np.float32), (227,227))
    
    # Reshape as needed to feed into model
    img = img.reshape((1,227,227,3))

    # Run the session and calculate the class probability
    probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
    
    print(i, probs, np.argmax(probs), labels[i])

sess.close()