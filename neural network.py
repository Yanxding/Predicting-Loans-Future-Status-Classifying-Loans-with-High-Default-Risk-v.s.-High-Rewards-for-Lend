# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 08:27:17 2019

@author: R.C.Weinstein
"""

# Machine Learning Final Project: Loan Status Prediction based on Tensorflow neural network
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import precision_recall_fscore_support
# Data Processing
data_raw = pd.read_csv('C:\\Users\\R.C.Weinstein\\Downloads\\resampled_training.csv', sep = '\t')
data_raw.columns.values.tolist()
data_raw = data_raw.drop(columns = 'Unnamed: 0')
data_test_x = pd.read_csv('C:\\Users\\R.C.Weinstein\\Downloads\\test_x.csv', sep = '\t')
data_test_y = pd.read_csv('C:\\Users\\R.C.Weinstein\\Downloads\\test_y.csv', sep = '\t')
data_test_x = data_test_x.drop(columns = 'Unnamed: 0')
data_test_y = data_test_y.drop(columns = ['Unnamed: 0', 'x'])
data_test_y[np.array(data_test_y == 4)] = 3
data_test_y[np.array(data_test_y == 5)] = 4
# training features and training labels Construction
data_raw_arr = np.array(data_raw)
traindata = data_raw_arr[:, :106]
trainlabel = data_raw_arr[:, -1]
trainlabel[trainlabel == 4] = 3
trainlabel[trainlabel == 5] = 4
testdata = np.array(data_test_x)
testlabel = np.array(data_test_y)

def label_change(before_label):
    label_num=len(before_label)
    change_arr=np.zeros((label_num,4))
    for i in range(label_num):
        change_arr[i,int(before_label[i] - 1)] = 1
    return change_arr

trainlabel = label_change(trainlabel)
testlabel = label_change(testlabel)
# Tensorflow preparations
## Defining the input and output
INPUT_NODE = np.shape(traindata)[1]
OUTPUT_NODE = 4
## Defining Hidden layer: the number of nodes and the number of layers
#Optimal Nodes:
LAYER1_NODE = 250
LAYER2_NODE = 400
LAYER3_NODE = 150
## Defining learning rate, stepsize, decay rate, etc..
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 2000
MOVING_AVERAGE_DECAY = 0.99
## Defining forward propagation function
def forward_propagation(input_tensor, avg_class, w1, b1, w2, b2, w3, b3, w4, b4):#, w5, b5
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
        layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
#        layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)
        return tf.matmul(layer3, w4) + b4
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(w1)) + avg_class.average(b1))
        layer2 = tf.nn.relu(tf.matmul(layer1, avg_class.average(w2)) + avg_class.average(b2))
        layer3 = tf.nn.relu(tf.matmul(layer2, avg_class.average(w3)) + avg_class.average(b3))
#        layer4 = tf.nn.relu(tf.matmul(layer3, avg_class.average(w4)) + avg_class.average(b4))
        return tf.matmul(layer3, avg_class.average(w4)) + avg_class.average(b4)
## Defining the train function
def train():
    x = tf.placeholder(tf.float32, shape = [None, INPUT_NODE], name = 'x-input')
    y_ = tf.placeholder(tf.float32, shape = [None, OUTPUT_NODE], name = 'y-input')
    # Initializing weights and bias one by one (Use random number here)
    w1 = tf.Variable(tf.truncated_normal(shape = [INPUT_NODE, LAYER1_NODE], stddev = 0.1))
    b1 = tf.Variable(tf.constant(0.1, shape = [LAYER1_NODE]))
    
    w2 = tf.Variable(tf.truncated_normal(shape = [LAYER1_NODE, LAYER2_NODE], stddev = 0.1))
    b2 = tf.Variable(tf.constant(0.1, shape = [LAYER2_NODE]))
    
    w3 = tf.Variable(tf.truncated_normal(shape = [LAYER2_NODE, LAYER3_NODE], stddev = 0.1))
    b3 = tf.Variable(tf.constant(0.1, shape = [LAYER3_NODE]))
    
    w4 = tf.Variable(tf.truncated_normal(shape = [LAYER3_NODE, OUTPUT_NODE], stddev = 0.1))
    b4 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))
#    
#    w5 = tf.Variable(tf.truncated_normal(shape = [LAYER4_NODE, OUTPUT_NODE], stddev = 0.1))
#    b5 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))
#    
    # Start inference:
    y = forward_propagation(x, None, w1, b1, w2, b2, w3, b3, w4, b4)#, w5, b5
    
    # Adding exponential moving average
    global_step = tf.Variable(0, trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    
    # Output averaged inference
    average_y = forward_propagation(x, variable_averages, w1, b1, w2, b2, w3, b3, w4, b4)#, w5, b5
    
    # Defining cross-entropy and loss function
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.arg_max(y_, 1))
    cross_entrip_mean = tf.reduce_mean(cross_entropy)
    
    # Regularized weights
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(w1) + regularizer(w2) + regularizer(w3) + regularizer(w4) #+ regularizer(w5)
    loss = cross_entrip_mean + regularization
    
    # Defining decling learning rate
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 900, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    train_op = tf.group(train_step, variable_averages_op)
    
    # Accu
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_feed = {x: traindata, y_: trainlabel}
        test_feed = {x: testdata, y_: testlabel}
        
        for i in range(TRAINING_STEPS):
            if i%10 == 0:
                train_acc = sess.run(accuracy, feed_dict = train_feed)
                print((i, train_acc))
            sess.run(train_op, feed_dict = train_feed)
        test_acc = sess.run(accuracy, feed_dict = test_feed)
        print((TRAINING_STEPS, test_acc))
        return sess.run(average_y, feed_dict = test_feed)

nodes_iter = [30, 60, 90, 120, 150, 200, 250, 300, 400]
accu_nodes_iter = []
for i in nodes_iter:
    accu_nodes_iter.append(train(i))
    
nodes_iter = [30, 60, 90, 120, 150, 200, 250, 300, 400]
accu_nodes_iter_2 = []
for i in nodes_iter:
    accu_nodes_iter_2.append(train(i))

nodes_iter = [30, 60, 90, 120, 150, 200, 250]#, 300, 400]
accu_nodes_iter_3 = []
for i in nodes_iter:
    accu_nodes_iter_3.append(train(i))


nodes_iter = [200, 250, 300, 400]
accu_nodes_iter_4 = []
for i in nodes_iter:
    accu_nodes_iter_4.append(train(i))
    
plt.plot(nodes_iter, np.array(accu_nodes_iter_4)[:,0][:,0], 'b-x', label = 'Charged Off')
plt.plot(nodes_iter, np.array(accu_nodes_iter_4)[:,0][:,1], 'r-x', label = 'Fully Paid')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy of each Class')
plt.title('Four Hidden Layers Neural Network Performance')
plt.legend()
    
plt.plot(nodes_iter, np.array(accu_nodes_iter_4)[:,0][:,2], 'b-x', label = 'Late 0-30 Days')
plt.plot(nodes_iter, np.array(accu_nodes_iter_4)[:,0][:,3], 'r-x', label = 'Late 31-120 Days')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy of each Class')
plt.title('Four Hidden Layers Neural Network Performance')
plt.legend()
    
optimal_nodes = [250, 400, 150, 150]
nodes_cor_index = [6, 8, 4, 4]
hidden_layer_num = [1,2,3,4]
acc_hidde_layer = np.array((accu_nodes_iter[6], 
                            accu_nodes_iter_2[8], 
                            accu_nodes_iter_3[4], 
                            accu_nodes_iter_4[4]))
plt.plot(hidden_layer_num, acc_hidde_layer[:,0][:,0], '-x', label = 'Charged Off')
plt.plot(hidden_layer_num, acc_hidde_layer[:,0][:,1], '-x', label = 'Fully Paid')
plt.plot(hidden_layer_num, acc_hidde_layer[:,0][:,2], '-x', label = 'Late 0-30 Days')
plt.plot(hidden_layer_num, acc_hidde_layer[:,0][:,3], '-x', label = 'Late 31-120 Days')
plt.xlabel('Number of Hidden Layer(s)')
plt.ylabel('Accuracy of each Class')
plt.legend()
starttime = datetime.datetime.now()
model_final = train()
endtime = datetime.datetime.now()
print (endtime - starttime).seconds

pred_final = []
for i in range(np.shape(model_final)[0]):
    pred_final.append(np.argmax(model_final[i]) + 1)
precision_recall_fscore_support(np.array(data_test_y), pred_final)


