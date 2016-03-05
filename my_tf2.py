import tensorflow as tf
import numpy as np
import os
from PIL import Image
cur_dir = os.getcwd()

def modify_image(image):
    #resized = tf.image.resize_images(image, 180, 180, 3)
    image.set_shape([32,32,3])
    flipped_images = tf.image.flip_up_down(image)
    return flipped_images

def read_image(filename_queue):
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    image = tf.image.decode_jpeg(value)
    return key,image
file_name=""
filenames_positive=[]
filenames_negative=[]
filenames=[]
def inputs():
    for i in range(1,126):
      
      if i<76:
       file_name="/home/tanvir/work/positive/standard_"+str(i)+".jpg"
       filenames_positive.append(file_name)
       
       j=1
      else:
       file_name="/home/tanvir/work/negative/standard_"+str(j)+".jpg"
       filenames_negative.append(file_name)
       
       j=j+1
    
    np.random.shuffle(filenames_positive)
    np.random.shuffle(filenames_negative)
    
    for i in range(0,75):
      filenames.append(filenames_positive[i])
    for i in range(0,50):
      filenames.append(filenames_negative[i])  
    
    #print(filenames)
    filename_queue = tf.train.string_input_producer(filenames)
    filename,read_input = read_image(filename_queue)
    reshaped_image = modify_image(read_input)
    
    reshaped_image = tf.cast(reshaped_image, tf.float32)
    
    
    return reshaped_image

def weight_variable(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial,name=name)

def bias_variable(shape,name):
  initial = tf.constant(1.0, shape=shape)
  return tf.Variable(initial,name=name)

def conv2d(x, W,name):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',name=name)

def max_pool_2x2(x,name):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME',name=name)

#x = tf.placeholder(tf.float32, shape=[None,32,32,3])
#y_ = tf.placeholder(tf.float32, shape=[None, 1])
with tf.Graph().as_default():
 image=inputs()
 
 image_batch=tf.train.batch([image],batch_size=150)
 label_batch_pos=tf.train.batch([tf.constant([0,1])],batch_size=75)
 label_batch_neg=tf.train.batch([tf.constant([1,0])],batch_size=50)
 label_batch=tf.concat(0,[label_batch_pos,label_batch_neg])
 
 W_conv1 = weight_variable([5, 5, 3, 10],"W_conv1")
 b_conv1 = bias_variable([10],"b_conv1")
 
 image_4d = tf.reshape(image, [-1,32,32,3])
 
 h_conv1 = tf.nn.relu(conv2d(image_4d, W_conv1,"conv2d1") + b_conv1)
 h_pool1 = max_pool_2x2(h_conv1,"max_pool1")
 
 W_conv2 = weight_variable([5, 5, 10, 20],"W_conv2")
 b_conv2 = bias_variable([20],"b_conv2")
 
 h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,"conv2d2") + b_conv2)
 h_pool2 = max_pool_2x2(h_conv2,"max_pool2")
 
 W_fc1 = weight_variable([8 * 8 * 20, 30],"W_fc1")
 b_fc1 = bias_variable([30],"b_fc1")
 
 h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*20])
 h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
 #keep_prob = tf.placeholder(tf.float32)
 h_fc1_drop = tf.nn.dropout(h_fc1, 0.8)
 
 W_fc2 = weight_variable([30, 2],"W_fc2")
 b_fc2 = bias_variable([2],"b_fc2")
 

 with tf.name_scope("O/P") as scope:
  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
 tf.histogram_summary("W_conv1",W_conv1) 
 tf.histogram_summary("b_conv1",b_conv1)
 tf.histogram_summary("y_conv",y_conv)

 with tf.name_scope("x_ent") as scope:
  cross_entropy = -tf.reduce_sum(tf.cast(label_batch,tf.float32)*tf.log(y_conv+1e-9))
  tf.scalar_summary("cross_entropy",cross_entropy)
 
 #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
 #      y_conv, tf.cast(label_batch,tf.float32), name='cross_entropy_per_example')
 with tf.name_scope("train") as scope:
  train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
 
 
 
 with tf.name_scope("test") as scope:
  correct_prediction=tf.equal(tf.argmax(y_conv,1), tf.argmax(tf.cast(label_batch,tf.float32),1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary("accuracy",accuracy)
 
 sess = tf.Session()
 merged=tf.merge_all_summaries()
 writer = tf.train.SummaryWriter("/home/tanvir/logs",
                                sess.graph.as_graph_def())
 init = tf.initialize_all_variables()
 
 sess.run(init)
 
 tf.train.start_queue_runners(sess=sess)
 
 for i in range(20000):
  result=sess.run([merged,accuracy])
  summary_str=result[0]
  acc=result[1]
  writer.add_summary(summary_str,i) 
  train_step.run(session=sess)
  print(sess.run(accuracy))

#print(sess.run(label_batch))






