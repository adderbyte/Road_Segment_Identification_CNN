import os
import sys
import matplotlib.image as mpimg
from PIL import Image

import numpy as np
import tensorflow as tf


class TensorFlowTrainer:
    """Doc"""

    def __init__(self):
        """Load the model"""

        self.session = None

        self.train_dir = "..\\tensorflow"
        self.training_size = 100
        self.batch_size = 64
        self.img_patch_size = 16
        self.num_channels = 3
        self.num_labels = 2
        self.seed = 66478
        self.pixel_depth = 255

        self.data_dir = '..\\training\\'
        self.train_data_filename = self.data_dir + 'images\\'
        self.train_labels_filename = self.data_dir + 'groundtruth\\'

        self.train_data_node = None
        self.train_labels_node = None
        self.train_all_data_node = None
        self.conv1_weights = None
        self.conv1_biases = None
        self.conv2_weights = None
        self.conv2_biases = None
        self.fc1_weights = None
        self.fc1_biases = None
        self.fc2_weights = None
        self.fc2_biases = None

        # Extract it into numpy arrays.
        self.train_data = self.extract_data(self.train_data_filename, self.training_size)
        self.train_labels = self.extract_labels(self.train_labels_filename, self.training_size)
        self.train_size = self.train_labels.shape[0]

    def set_training_size(self, size):
        self.training_size = size

    def set_batch_size(self, size):
        self.batch_size = size

    def set_img_patch_size(self, size):
        self.img_patch_size = size

    def load_data(self):
        c0 = 0
        c1 = 0
        for i in range(len(self.train_labels)):
            if self.train_labels[i][0] == 1:
                c0 += 1
            else:
                c1 += 1
        print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

        print('Balancing training data...')
        min_c = min(c0, c1)
        idx0 = [i for i, j in enumerate(self.train_labels) if j[0] == 1]
        idx1 = [i for i, j in enumerate(self.train_labels) if j[1] == 1]
        new_indices = idx0[0:min_c] + idx1[0:min_c]
        print(len(new_indices))
        print(self.train_data.shape)
        self.train_data = self.train_data[new_indices, :, :, :]
        self.train_labels = self.train_labels[new_indices]

        self.train_size = self.train_labels.shape[0]

        c0 = 0
        c1 = 0
        for i in range(len(self.train_labels)):
            if self.train_labels[i][0] == 1:
                c0 += 1
            else:
                c1 += 1
        print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        self.train_data_node = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.img_patch_size, self.img_patch_size, self.num_channels))
        self.train_labels_node = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.num_labels))
        self.train_all_data_node = tf.constant(self.train_data)

        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when when we call:
        # {tf.initialize_all_variables().run()}
        self.conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, self.num_channels, 32],  # 5x5 filter, depth 32.
                                stddev=0.1,
                                seed=self.seed))
        self.conv1_biases = tf.Variable(tf.zeros([32]))
        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
                                stddev=0.1,
                                seed=self.seed))
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([int(self.img_patch_size / 4 * self.img_patch_size / 4 * 64), 512],
                                stddev=0.1,
                                seed=self.seed))
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
        self.fc2_weights = tf.Variable(
            tf.truncated_normal([512, self.num_labels],
                                stddev=0.1,
                                seed=self.seed))
        self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.num_labels]))

    def train(self, num_epochs=10, save_predictionbs=False, restore=False):

        # Training computation: logits + cross-entropy loss.
        logits = self.model(self.train_data_node, True)  # BATCH_SIZE*NUM_LABELS
        # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits, self.train_labels_node))
        tf.scalar_summary('loss', loss)

        all_params_node = [self.conv1_weights, self.conv1_biases, self.conv2_weights, self.conv2_biases,
                           self.fc1_weights,
                           self.fc1_biases, self.fc2_weights, self.fc2_biases]
        all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights',
                            'fc1_biases', 'fc2_weights', 'fc2_biases']
        all_grads_node = tf.gradients(loss, all_params_node)
        all_grad_norms_node = []
        for i in range(0, len(all_grads_node)):
            norm_grad_i = tf.global_norm([all_grads_node[i]])
            all_grad_norms_node.append(norm_grad_i)
            tf.scalar_summary(all_params_names[i], norm_grad_i)

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                        tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch * self.batch_size,  # Current index into the dataset.
            self.train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        tf.scalar_summary('learning_rate', learning_rate)

        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.0).minimize(loss, global_step=batch)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=batch)

        # Predictions for the minibatch, validation set and test set.
        train_prediction = tf.nn.softmax(logits)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a local session to run this computation.
        with tf.Session() as s:
            self.session = s  # later on for the predictions

            if restore:
                # Restore variables from disk.
                saver.restore(s, self.train_dir + "\\model.ckpt")
                print("Model restored.")

            else:
                # Run all the initializers to prepare the trainable parameters.
                tf.global_variables_initializer().run()

                print('Initialized!')
                # Loop through training steps.
                print('Total number of iterations = ' + str(int(num_epochs * self.train_size / self.batch_size)))

                training_indices = range(self.train_size)

                for iepoch in range(num_epochs):

                    # Permute training indices
                    perm_indices = np.random.permutation(training_indices)

                    for step in range(int(self.train_size / self.batch_size)):

                        offset = (step * self.batch_size) % (self.train_size - self.batch_size)
                        batch_indices = perm_indices[offset:(offset + self.batch_size)]

                        # Compute the offset of the current minibatch in the data.
                        # Note that we could use better randomization across epochs.
                        batch_data = self.train_data[batch_indices, :, :, :]
                        batch_labels = self.train_labels[batch_indices]
                        # This dictionary maps the batch data (as a numpy array) to the
                        # node in the graph is should be fed to.
                        feed_dict = {self.train_data_node: batch_data,
                                     self.train_labels_node: batch_labels}

                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                        if step % 1000 == 0:
                            print('Epoch %.2f' % (float(step) * self.batch_size / self.train_size))
                            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                            print('Minibatch error: %.1f%%' % self.error_rate(predictions, batch_labels))

                            sys.stdout.flush()

                    # Save the variables to disk.
                    save_path = saver.save(s, self.train_dir + "\\model.ckpt")
                    print("Model saved in file: %s" % save_path)

            print("Running prediction on training set")
            prediction_training_dir = "..\\training\\predictions\\"
            if not os.path.isdir(prediction_training_dir):
                os.mkdir(prediction_training_dir)

            predictions_groundtruth = np.array([])
            labels_groundtruth = np.array([])
            for i in range(1, self.training_size + 1):
                gimg, pimg = self.get_prediction_with_groundtruth(self.train_data_filename, i)

                # Keep the groundtruth
                predictions_groundtruth = np.append(predictions_groundtruth, np.reshape(gimg, (-1, 1)))

                imageid = "satImage_%.3d" % i
                image_filename = self.train_labels_filename + imageid + ".png"
                img = mpimg.imread(image_filename)
                labels_groundtruth = np.append(labels_groundtruth, np.reshape(img, (-1, 1)))

                if save_predictionbs:
                    Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
                    oimg = self.get_prediction_with_overlay(self.train_data_filename, i)
                    oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")

            # Compute the train set error
            print('Precision', np.sum(labels_groundtruth == predictions_groundtruth) / len(labels_groundtruth))

    def model(self, data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            self.conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                             self.conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

        # Uncomment these lines to check the size of each layer
        # print('data ' + str(data.get_shape()))
        # print('conv ' + str(conv.get_shape()))
        # print('relu ' + str(relu.get_shape()))
        # print('pool ' + str(pool.get_shape()))
        # print('pool2 ' + str(pool2.get_shape()))

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=self.seed)

        out = tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

        return out

    def get_image_summary(self, img, idx=0):
        v = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(v)
        v = v - min_value
        max_value = tf.reduce_max(v)
        v /= max_value * self.pixel_depth
        v = tf.reshape(v, (img_w, img_h, 1))
        v = tf.transpose(v, (2, 0, 1))
        v = tf.reshape(v, (-1, img_w, img_h, 1))
        return v

    def get_prediction(self, img):
        data = np.asarray(self.img_crop(img, self.img_patch_size, self.img_patch_size))
        data_node = tf.constant(data)
        output = tf.nn.softmax(self.model(data_node))
        output_prediction = self.session.run(output)
        img_prediction = self.label_to_img(img.shape[0], img.shape[1], self.img_patch_size, self.img_patch_size,
                                           output_prediction)

        return img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(self, filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = self.get_prediction(img)
        cimg = self.concatenate_images(img, img_prediction)

        return img_prediction, cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(self, filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = self.get_prediction(img)
        oimg = self.make_img_overlay(img, img_prediction)

        return oimg

    # Convert array of labels to an image
    @staticmethod
    def label_to_img(imgwidth, imgheight, w, h, labels):
        array_labels = np.zeros([imgwidth, imgheight])
        idx = 0
        for i in range(0, imgheight, h):
            for j in range(0, imgwidth, w):
                if labels[idx][0] > 0.5:
                    l = 0
                else:
                    l = 1
                array_labels[j:j + w, i:i + h] = l
                idx += 1
        return array_labels

    def concatenate_images(self, img, gt_img):
        n_channels = len(gt_img.shape)
        w = gt_img.shape[0]
        h = gt_img.shape[1]
        if n_channels == 3:
            cimg = np.concatenate((img, gt_img), axis=1)
        else:
            gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
            gt_img8 = self.img_float_to_uint8(gt_img)
            gt_img_3c[:, :, 0] = gt_img8
            gt_img_3c[:, :, 1] = gt_img8
            gt_img_3c[:, :, 2] = gt_img8
            img8 = self.img_float_to_uint8(img)
            cimg = np.concatenate((img8, gt_img_3c), axis=1)
        return cimg

    def make_img_overlay(self, img, predicted_img):
        w = img.shape[0]
        h = img.shape[1]
        color_mask = np.zeros((w, h, 3), dtype=np.uint8)
        color_mask[:, :, 0] = predicted_img * self.pixel_depth

        img8 = self.img_float_to_uint8(img)
        background = Image.fromarray(img8, 'RGB').convert("RGBA")
        overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
        new_img = Image.blend(background, overlay, 0.2)
        return new_img

    def img_float_to_uint8(self, img):
        rimg = img - np.min(img)
        rimg = (rimg / np.max(rimg) * self.pixel_depth).round().astype(np.uint8)
        return rimg

    def extract_data(self, filename, num_images):
        """
        Extract the images into a 4D tensor [image index, y, x, channels].
        Values are rescaled from [0, 255] down to [-0.5, 0.5].
        """
        imgs = []
        for i in range(1, num_images + 1):
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
            if os.path.isfile(image_filename):
                # print ('Loading ' + image_filename)
                img = mpimg.imread(image_filename)
                imgs.append(img)
            else:
                print('File ' + image_filename + ' does not exist')

        num_images = len(imgs)
        # img_width = imgs[0].shape[0]
        # img_height = imgs[0].shape[1]
        # N_PATCHES_PER_IMAGE = (img_width / self.img_patch_size) * (img_height / self.img_patch_size)

        img_patches = [self.img_crop(imgs[i], self.img_patch_size, self.img_patch_size) for i in range(num_images)]
        data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

        return np.asarray(data)

    def extract_labels(self, filename, num_images):
        """Extract the labels into a 1-hot matrix [image index, label index]."""
        gt_imgs = []
        for i in range(1, num_images + 1):
            imageid = "satImage_%.3d" % i
            image_filename = filename + imageid + ".png"
            if os.path.isfile(image_filename):
                # print ('Loading ' + image_filename)
                img = mpimg.imread(image_filename)
                gt_imgs.append(img)
            else:
                print('File ' + image_filename + ' does not exist')

        num_images = len(gt_imgs)
        gt_patches = [self.img_crop(gt_imgs[i], self.img_patch_size, self.img_patch_size) for i in range(num_images)]
        data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
        labels = np.asarray([self.value_to_class(np.mean(data[i])) for i in range(len(data))])

        # Convert to dense 1-hot representation.
        return labels.astype(np.float32)

    # Extract patches from a given image
    @staticmethod
    def img_crop(im, w, h):
        list_patches = []
        imgwidth = im.shape[0]
        imgheight = im.shape[1]
        is_2d = len(im.shape) < 3
        for i in range(0, imgheight, h):
            for j in range(0, imgwidth, w):
                if is_2d:
                    im_patch = im[j:j + w, i:i + h]
                else:
                    im_patch = im[j:j + w, i:i + h, :]
                list_patches.append(im_patch)
        return list_patches

    # Assign a label to a patch v
    @staticmethod
    def value_to_class(v):
        foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
        df = np.sum(v)
        if df > foreground_threshold:
            return [0, 1]
        else:
            return [1, 0]

    @staticmethod
    def error_rate(predictions, labels):
        """Return the error rate based on dense predictions and 1-hot labels."""
        return 100.0 - (
            100.0 *
            np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])
