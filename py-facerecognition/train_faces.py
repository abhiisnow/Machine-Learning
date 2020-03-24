import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

my_faces_path = './my_faces'
other_faces_path = './other_faces'
size = 64

imgs = []
labs = []


def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

detector = dlib.get_frontal_face_detector()

def readData(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            print(filename)
            cv2.namedWindow('image',0)
            cv2.imshow('image',img)

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_img, 1)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1,x2:y2]
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                face = cv2.resize(face, (size,size))

            top,bottom,left,right = getPaddingSize(face)
            face = cv2.copyMakeBorder(face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            face = cv2.resize(face, (h, w))
            cv2.namedWindow('processed',0)
            cv2.imshow('processed',face)
            cv2.waitKey(1)
            
            imgs.append(face)
            labs.append(path)
    cv2.destroyAllWindows()

readData(my_faces_path)
readData(other_faces_path)
imgs = np.array(imgs)
labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])
train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0,100))
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
batch_size = 100
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    W1 = weightVariable([3,3,3,32]) 
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    pool1 = maxPool(conv1)
    drop1 = dropout(pool1, keep_prob_5)

    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    Wout = weightVariable([512,2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

def cnnTrain():
    out = cnnLayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for n in range(10):
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                print(n*num_batch+i,' loss:', loss)

                if (n*num_batch+i) % 10 == 0:
                    acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                    print(n*num_batch+i, ' acc:', acc)
                    if acc > 0.99 and n > 2:
                        saver.save(sess, './train_faces.model', global_step=n*num_batch+i)
                        sys.exit(0)
        print('accuracy less 0.99, exited!')


cnnTrain()
