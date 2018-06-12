'''
	Python 3.5
'''

import os
import numpy as np
import time

import tensorflow as tf
from alexnet import AlexNet #
from datetime import datetime

from tensorflow.contrib.data import Iterator
from datagenerator import ImageDataGenerator
from tensorflow.contrib.data import Iterator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
	training parameters
'''

learning_rate = 0.001
# num_epochs = 10
num_epochs = 2
batch_size = 32
# Network params
dropout_rate = 0.5
num_classes = 2
# How often we want to write the tf.summary data to disk
display_step = 20
save_step=2000

# to-do:
train_file=''
val_file=''

'''
	Data IO: tf.data
'''
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()
    
# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

print(train_batches_per_epoch, val_batches_per_epoch)

'''
	Graph Def
'''
# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, skip_layer=None)

# Link variable to model output
score = model.fc8

# trainable variables
var_list = [v for v in tf.trainable_variables()]

# Op for calculating the loss
# cross entropy loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))
# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name.replace(":", "_") + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name.replace(":", "_"), var)

# Add the loss to summary
tf.summary.scalar('Loss', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('Accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()


'''
	Train
'''

# to-do:
newmodel=True
pre_model=''

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path + '/' + dirname + '/')

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Start Tensorflow session
with tf.Session() as sess:

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the Imagenet pretrained weights into the non-trainable layer
    # Finetune the whole network
    
    if newmodel:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess, trainablev=True, layersList=['fc8'])
    else:
        saver.restore(sess, pre_model)
    
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))
    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            # ---------- Fetch Data ------------
            img_batch, label_batch = sess.run(next_batch)
            # ---------- Fetch Data End ------------
            
            # And run the training op
            # ---------- Train ------------
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})
            # ---------- Train End ------------
            
            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)
                
            if step % save_step == 0:
                # checkpoint_name = os.path.join(checkpoint_path,
                #                dirname + '-TMPmodel_epoch'+str(epoch+1) + '-' + \
                #                str(datetime.now()).replace(' ', '-').split('.')[0] + \
                #                '.ckpt')

                # to-do:
                checkpoint_name=''
                save_path = saver.save(sess, checkpoint_name)

        # Validate the model on the entire validation set
        # ---------- Validation ------------
        
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
            
        test_acc /= test_count
        
        # ---------- Validation End ------------
        
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        # checkpoint_name = os.path.join(checkpoint_path,
        #                                dirname + '-model_epoch'+str(epoch+1) + '-' + \
        #                                str(datetime.now()).replace(' ', '-').split('.')[0] + \
        #                                '.ckpt')
        # to-do:
        checkpoint_name=''
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
