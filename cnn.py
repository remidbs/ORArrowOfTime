import tensorflow as tf
import numpy as np
from scipy.misc import imread
import sys
import os
import pandas as pd


def get_training_images(nb_training_examples):
    x = []
    y = []

    video_paths = os.listdir("Samples_resized/")
    it = 0
    for video_path in video_paths:
        if(video_path == ".DS_Store"):
            continue
        it += 1
        if(it > nb_training_examples):
            break


        a = imread("Samples_resized/"+video_path+"/a.png")[:,:,:3].reshape((227,227,3,1))
        b = imread("Samples_resized/"+video_path+"/b.png")[:,:,:3].reshape((227, 227, 3, 1))
        c = imread("Samples_resized/"+video_path+"/c.png")[:,:,:3].reshape((227, 227, 3, 1))
        d = imread("Samples_resized/"+video_path+"/d.png")[:, :,:3].reshape((227, 227, 3, 1))
        e = imread("Samples_resized/"+video_path+"/e.png")[:, :, :3].reshape((227, 227, 3, 1))

        #True tuples
        x.append(np.concatenate([b, c, d], axis=3))
        y.append([1, 0])
        x.append(np.concatenate([d, c, b], axis=3))
        y.append([1, 0])

        #False tuples
        x.append(np.concatenate([b, a, d], axis=3))
        y.append([0, 1])
        x.append(np.concatenate([d, a, b], axis=3))
        y.append([0, 1])
        x.append(np.concatenate([b, e, d], axis=3))
        y.append([0, 1])
        x.append(np.concatenate([d, e, b], axis=3))
        y.append([0, 1])

    return np.asarray(x), np.asarray(y)
    


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


# Create the model
x = tf.placeholder(tf.float32, [None,227,227,3,3])
x_image = tf.reshape(x, [-1,227,227,3,3])
x_image1 = x_image[:,:,:,:,0]
x_image2 = x_image[:,:,:,:,1]
x_image3 = x_image[:,:,:,:,2]
net_data = np.load("bvlc_alexnet.npy").item()

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0], name='conv1')
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in1 = conv(x_image1, conv1W, conv1b, k_h, k_w,
                c_o, s_h, s_w, padding="SAME", group=1)
conv11 = tf.nn.relu(conv1_in1)
conv1_in2 = conv(x_image2, conv1W, conv1b, k_h, k_w,
                c_o, s_h, s_w, padding="SAME", group=1)
conv12 = tf.nn.relu(conv1_in2)
conv1_in3 = conv(x_image3, conv1W, conv1b, k_h, k_w,
                c_o, s_h, s_w, padding="SAME", group=1)
conv13 = tf.nn.relu(conv1_in3)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn11 = tf.nn.local_response_normalization(conv11,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
lrn12 = tf.nn.local_response_normalization(conv12,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
lrn13 = tf.nn.local_response_normalization(conv13,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool11 = tf.nn.max_pool(lrn11, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding)
maxpool12 = tf.nn.max_pool(lrn12, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding)
maxpool13 = tf.nn.max_pool(lrn13, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in1 = conv(maxpool11, conv2W, conv2b, k_h, k_w, c_o,
                s_h, s_w, padding="SAME", group=group)
conv21 = tf.nn.relu(conv2_in1)
conv2_in2 = conv(maxpool12, conv2W, conv2b, k_h, k_w, c_o,
                s_h, s_w, padding="SAME", group=group)
conv22 = tf.nn.relu(conv2_in2)
conv2_in3 = conv(maxpool13, conv2W, conv2b, k_h, k_w, c_o,
                s_h, s_w, padding="SAME", group=group)
conv23 = tf.nn.relu(conv2_in3)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn21 = tf.nn.local_response_normalization(conv21,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
lrn22 = tf.nn.local_response_normalization(conv22,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
lrn23 = tf.nn.local_response_normalization(conv23,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool21 = tf.nn.max_pool(lrn21, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding)
maxpool22 = tf.nn.max_pool(lrn22, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding)
maxpool23 = tf.nn.max_pool(lrn23, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in1 = conv(maxpool21, conv3W, conv3b,
                k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv31 = tf.nn.relu(conv3_in1)
conv3_in2 = conv(maxpool22, conv3W, conv3b,
                k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv32 = tf.nn.relu(conv3_in2)
conv3_in3 = conv(maxpool23, conv3W, conv3b,
                k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv33 = tf.nn.relu(conv3_in3)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in1 = conv(conv31, conv4W, conv4b, k_h, k_w,
                c_o, s_h, s_w, padding="SAME", group=group)
conv41 = tf.nn.relu(conv4_in1)
conv4_in2 = conv(conv32, conv4W, conv4b, k_h, k_w,
                c_o, s_h, s_w, padding="SAME", group=group)
conv42 = tf.nn.relu(conv4_in2)
conv4_in3 = conv(conv33, conv4W, conv4b, k_h, k_w,
                c_o, s_h, s_w, padding="SAME", group=group)
conv43 = tf.nn.relu(conv4_in3)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in1 = conv(conv41, conv5W, conv5b, k_h, k_w, c_o,
                s_h, s_w, padding="SAME", group=group)
conv51 = tf.nn.relu(conv5_in1)
conv5_in2 = conv(conv42, conv5W, conv5b, k_h, k_w, c_o,
                s_h, s_w, padding="SAME", group=group)
conv52 = tf.nn.relu(conv5_in2)
conv5_in3 = conv(conv43, conv5W, conv5b, k_h, k_w, c_o,
                s_h, s_w, padding="SAME", group=group)
conv53 = tf.nn.relu(conv5_in3)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool51 = tf.nn.max_pool(conv51, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding)
maxpool52 = tf.nn.max_pool(conv52, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding)
maxpool53 = tf.nn.max_pool(conv53, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc61 = tf.nn.relu_layer(tf.reshape(maxpool51, [-1, int(np.prod(maxpool51.get_shape()[1:]))]),
                       fc6W, fc6b)
fc62 = tf.nn.relu_layer(tf.reshape(maxpool52, [-1, int(np.prod(maxpool52.get_shape()[1:]))]),
                       fc6W, fc6b)
fc63 = tf.nn.relu_layer(tf.reshape(maxpool53, [-1, int(np.prod(maxpool53.get_shape()[1:]))]),
                       fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc71 = tf.nn.relu_layer(fc61, fc7W, fc7b)
fc72 = tf.nn.relu_layer(fc62, fc7W, fc7b)
fc73 = tf.nn.relu_layer(fc63, fc7W, fc7b)
fc7  = tf.concat(1, [fc71,fc72,fc73])

#fc8
#fc(1000, relu=False, name='fc8')
#fc8W = tf.Variable(net_data["fc8"][0])
#fc8b = tf.Variable(net_data["fc8"][1])
#fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

#fc8
initial1 = tf.truncated_normal([fc7.get_shape()[1].value, 5], stddev=0.1)
Wfc8 = tf.Variable(initial1)
initial2 = tf.constant(0.1, shape=[5])
bfc8 = tf.Variable(initial2)
fc8 = tf.matmul(fc7, Wfc8) + bfc8

#output
initial3 = tf.truncated_normal([fc8.get_shape()[1].value, 2], stddev=0.1)
W = tf.Variable(initial3)
initial4 = tf.constant(0.1, shape=[2])
b = tf.Variable(initial4)
y = tf.matmul(fc8, W) + b
    

def train(sess, x_in, y_in):
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    sess.run(tf.global_variables_initializer())

    # Train
    batch_size = 1080
    for t in range(y_in.shape[0]/batch_size):
        print "batch nb",t,"out of ",y_in.shape[0]/batch_size
        sess.run(train_step, feed_dict={x: x_in[t*batch_size: (t+1)*batch_size], y_: y_in[t*batch_size: (t+1)*batch_size]})

    # Test trained model
    print "Testing neural network..."
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: x_in,
                                          y_: y_in}))
    

def persist(filepath, sess):
    saver = tf.train.Saver()
    saver.save(sess, filepath)

mode="restore"

x_in, y_in = get_training_images(np.inf)
if(mode == "train"):
    sess = tf.InteractiveSession()
    train(sess, x_in, y_in)
    persist("model-30-12",sess)
else:
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    new_saver = tf.train.import_meta_graph('model-30-12.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print(v.name)
    pred = sess.run(fc8, feed_dict={x: x_in})
    DF = pd.DataFrame(np.concatenate([np.concatenate([pred[i*6+j,:] for j in range(6)]) for i in range(pred.shape[0]/6)]))    
    DF["name"] = os.listdir("Samples_resized/")[1:]
    DF["label"] = DF.name.apply(lambda x : x[0] == 'F')
    DF.to_csv("features/features30-12.csv", index=None, header=None)
        
    
