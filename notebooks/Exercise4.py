import h5py
import numpy as np
import random
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use('Agg')
 

class Data:
  def __init__(self):
    with h5py.File("cell_data.h5", "r") as data:
      self.train_images = [data["/train_image_{}".format(i)][:] for i in range(28)]
      self.train_labels = [data["/train_label_{}".format(i)][:] for i in range(28)]
      self.test_images = [data["/test_image_{}".format(i)][:] for i in range(3)]
      self.test_labels = [data["/test_label_{}".format(i)][:] for i in range(3)]
    
    self.input_resolution = 300
    self.label_resolution = 116

    self.offset = (300 - 116) // 2

  def get_train_image_list_and_label_list(self):
    n = random.randint(0, len(self.train_images) - 1)
    x = random.randint(0, (self.train_images[n].shape)[1] - self.input_resolution - 1)
    y = random.randint(0, (self.train_images[n].shape)[0] - self.input_resolution - 1)
    image = self.train_images[n][y:y + self.input_resolution, x:x + self.input_resolution, :]

    x += self.offset
    y += self.offset
    label = self.train_labels[n][y:y + self.label_resolution, x:x + self.label_resolution]
    
    return [image], [label]

  def get_test_image_list_and_label_list(self):
    coord_list = [[0,0], [0, 116], [0, 232], 
                  [116,0], [116, 116], [116, 232],
                  [219,0], [219, 116], [219, 232]]
    
    image_list = []
    label_list = []
    
    for image_id in range(3):
      for y, x in coord_list:
        image = self.test_images[image_id][y:y + self.input_resolution, x:x + self.input_resolution, :]
        image_list.append(image)
        x += self.offset
        y += self.offset
        label = self.test_labels[image_id][y:y + self.label_resolution, x:x + self.label_resolution]
        label_list.append(label)
    return image_list, label_list



def crop_and_concat(x1, x2):
    """crop and concatenate the image
    """
    x2_rows = x2.get_shape().as_list()[1]
    x2_columns = x2.get_shape().as_list()[2]
    
    x1_crop = tf.image.resize_image_with_crop_or_pad(x1, x2_rows, x2_columns)
    return tf.concat([x1_crop, x2], 3) 
    
    
def predict(x):
    """predict returns prediction given input
    """
    x_image = tf.reshape(x, [-1, 300, 300, 1])
    conv1 = tf.layers.conv2d(x_image, filters=32, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    
    maxpool1 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2))
    
    conv3 = tf.layers.conv2d(maxpool1, filters=64, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    
    maxpool2 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2))
    
    conv5 = tf.layers.conv2d(maxpool2, filters=128, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(conv5, filters=128, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    
    maxpool3 = tf.layers.max_pooling2d(conv6, pool_size=(2,2), strides=(2,2))
    
    conv7 = tf.layers.conv2d(maxpool3, filters=256, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    conv8 = tf.layers.conv2d(conv7, filters=256, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    
    maxpool4 = tf.layers.max_pooling2d(conv8, pool_size=(2,2), strides=(2,2))
    
    conv9 = tf.layers.conv2d(maxpool4, filters=512, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    conv10 = tf.layers.conv2d(conv9, filters=512, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    
    conv2d_transpose1 = tf.layers.conv2d_transpose(conv10, filters=256, kernel_size=(2, 2), padding='VALID', strides=2)
    
    conv2d_transpose1_concat = crop_and_concat(conv8, conv2d_transpose1)    
    conv11 = tf.layers.conv2d(conv2d_transpose1_concat, filters=256, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    conv12 = tf.layers.conv2d(conv11, filters=256, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    
    conv2d_transpose2 = tf.layers.conv2d_transpose(conv12, filters=128, kernel_size=(2, 2), padding='VALID', strides=2)
    conv2d_transpose2_concat = crop_and_concat(conv6, conv2d_transpose2)
    
    conv13 = tf.layers.conv2d(conv2d_transpose2_concat, filters=128, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    conv14 = tf.layers.conv2d(conv13, filters=128, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    
    conv2d_transpose3 = tf.layers.conv2d_transpose(conv14, filters=64, kernel_size=(2, 2), padding='VALID', strides=2)
    conv2d_transpose3_concat = crop_and_concat(conv4, conv2d_transpose3)
    
    conv15 = tf.layers.conv2d(conv2d_transpose3_concat, filters=64, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    conv16 = tf.layers.conv2d(conv15, filters=64, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    
    conv2d_transpose4 = tf.layers.conv2d_transpose(conv16, filters=32, kernel_size=(2, 2), padding='VALID', strides=2)
    conv2d_transpose4_concat = crop_and_concat(conv2, conv2d_transpose4)
    
    conv17 = tf.layers.conv2d(conv2d_transpose4_concat, filters=32, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    conv18 = tf.layers.conv2d(conv17, filters=32, kernel_size=(3,3), strides=(1,1), padding='VALID', use_bias=True, activation=tf.nn.relu)
    
    output = tf.layers.conv2d(conv18, filters=2, kernel_size=(1,1), strides=(1,1), padding='VALID', use_bias=True)
    return output


def get_accuracy(y_predict, y_):
    """get segmentation accuracy using 'intersection over union' 
    """
    prediction = np.argmax(y_predict, axis=3)
    correct_cell_pixel = np.sum(prediction == y_)
    incorrect_cell_pixel = np.sum(prediction != y_)
    segmentation_accuracy = correct_cell_pixel / (correct_cell_pixel + 2 * incorrect_cell_pixel)
    return segmentation_accuracy


def plot_learning_curves(training_accuracy, test_accuracy):
    """plot and save learning curves using training and test accuracy
    """
    plt.plot(range(1,len(training_accuracy) + 1), training_accuracy, label='Training accuracy')
    plt.plot(range(1,len(test_accuracy) + 1), test_accuracy, label='Validation accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.xlabel('number of iterations')
    plt.ylabel('accuracy')
    plt.savefig('learning_curve.png', bbox_inches='tight')
    plt.show()


def plot_loss(loss):
    """plot and save loss during training
    """
    plt.plot(range(1,len(loss) + 1), loss)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.xlabel('number of iterations')
    plt.ylabel('training loss')
    plt.savefig('loss.png')
    plt.show()
    

def plot_images(original_image, given_image,  predicted_image, i):
    """plot original and segmented image
    """
    fig = plt.figure()
    a=fig.add_subplot(3,1,1)
    
    plt.axis('off')
    imgplot = plt.imshow(original_image.squeeze(),cmap='gray')
    a.set_title('Original Image')
    
    a=fig.add_subplot(3,1,2)
    
    plt.axis('off')
    imgplot = plt.imshow(given_image.squeeze(),cmap='gray')
    a.set_title('Given Segmentation')
    
    a=fig.add_subplot(3,1,3)
    imgplot = plt.imshow(predicted_image.squeeze(),cmap='gray')

    a.set_title('Predicted Segmentation')
    plt.axis('off')
    
    plt.savefig('image_%s.png' % (i))
    plt.show()

    
def train(d):
    """ Train network on the given data. """
    
    # Define placeholder for x
    x = tf.placeholder(tf.float32, [None, 300, 300, 1])
    # Define placeholder for y
    y_ = tf.placeholder(tf.int32, [None, 116, 116])
    
    # Predict given the data
    y_predict = predict(x)
    
    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_predict))
    train_step = tf.train.AdamOptimizer(0.0001, 0.95, 0.99).minimize(loss)
    
    #lists for holding required data
    loss_list = list()
    training_accuracy = list()
    test_accuracy = list()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(40000):
            
            train_batch_xs, train_batch_ys = d.get_train_image_list_and_label_list()
            test_batch_xs, test_batch_ys = d.get_test_image_list_and_label_list()
            output_train, _, batch_loss = sess.run([y_predict, train_step, loss],feed_dict={x: train_batch_xs, y_:train_batch_ys})
            
            output_train, batch_loss = sess.run([y_predict, loss],feed_dict={x: train_batch_xs, y_:train_batch_ys})
            output_test = sess.run(y_predict,feed_dict={x: test_batch_xs, y_:train_batch_ys})
            
            training_accuracy.append(get_accuracy(output_train, train_batch_ys))
            test_accuracy.append(get_accuracy(output_test, test_batch_ys))             
            loss_list.append(batch_loss)
        
        plot_learning_curves(training_accuracy, test_accuracy)
        plot_loss(loss_list)
        
        test_batch_xs, test_batch_ys = d.get_test_image_list_and_label_list()
        
        for i in range(3):
            original_image = np.array(test_batch_xs)[i,:,:]
            given_image = np.array(test_batch_ys)[i,:,:]
            prediction = sess.run(y_predict, feed_dict={x: test_batch_xs})
            predicted_image = np.argmax(prediction, axis=3)
            
            plot_images(original_image, given_image, predicted_image[i], i)

        
def main():
    d = Data()
    train(d)

if __name__ == "__main__":
    main()
