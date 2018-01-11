from __future__ import print_function
from pylab import *

import os
import struct
import matplotlib.pyplot as plt
import numpy as np


# comment this line out if not using cuda
# see https://github.com/cudamat/cudamat for docs
# import cudamat as cm


class RBM:
    def __init__(self, num_visible, num_hidden, persisten, with_cuda):
        self.cuda = with_cuda
        self.debug = True
        self.weights_dir = "/rbm_weights/"
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.examples_size = 0
        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a uniform distribution between -sqrt(6. / (num_hidden + num_visible))
        # and sqrt(6. / (num_hidden + num_visible)).
        # Reference: Understanding the difficulty of training deep feedforward
        # neural networks by Xavier Glorot and Yoshua Bengio
        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
            low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size=(num_visible, num_hidden)))

        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)
        self.save = persisten

    def _propability(self, x, training=False):
        """
        default activation function
        :param x: value to compute
        :return:
        """
        if self.cuda and training:
            return cm.CUDAMatrix(ones(x.shape)).divide(cm.exp(x.mult(-1.0)).add(1.0))
        return 1.0 / (1 + np.exp(-x))

    def _one_training_step(self, data, step_number, learning_rate, num_examples, bias=None, data_t=None):
        """
        Performs one single training step and ajust the weight based on the learning rate
        and the error rate
        :param data: training data
        :param step_number: current step number
        :param learning_rate: learning ratio
        :return: none
        """

        # compute hidden states -> visible -> hidden again and compare the difference -> add it to the weights

        if self.cuda:
            pos_hidden_activations = cm.dot(data, self.weights_cuda)
            pos_hidden_probs = self._propability(pos_hidden_activations, training=True)
            pos_hidden_probs.set_col_slice(0, 1, bias.copy())
            pos_associations = cm.dot(data_t, pos_hidden_probs)
            pos_hidden_states = pos_hidden_probs.greater_than(
                cm.CUDAMatrix(np.random.rand(num_examples, self.num_hidden + 1)))
            neg_visible_activations = cm.dot(pos_hidden_states, self.weights_cuda.transpose())
            neg_visible_probs = self._propability(neg_visible_activations, training=True)

            # fix bias unit
            neg_visible_probs.set_col_slice(0, 1, bias.copy())
            neg_hidden_activations = cm.dot(neg_visible_probs, self.weights_cuda)
            neg_hidden_probs = self._propability(neg_hidden_activations, training=True)
            # use propabilitys instead of actual states -> faster
            # Traverse the data to match the matrix dimensions
            neg_associations = cm.dot(neg_visible_probs.transpose(), neg_hidden_probs)
            # Update weights.
            self.weights_cuda.add(
                ((pos_associations.subtract(neg_associations)).divide(float(num_examples))).mult(
                    learning_rate))
            if self.debug:
                error = cm.sum(cm.sum((cm.pow(data.copy().subtract(neg_visible_probs), 2)), 1), 0).asarray()[0, 0]

        else:
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = self._propability(pos_hidden_activations, training=True)
            pos_hidden_probs[:, 0] = 1  # Fix the bias unit.
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)

            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
            # themselves, when computing associations.
            pos_associations = np.dot(data.T, pos_hidden_probs)
            # Reconstruct the visible units and sample again from the hidden units.
            # Backward computing phase, hidden -> visible
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._propability(neg_visible_activations, training=True)
            neg_visible_probs[:, 0] = 1  # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._propability(neg_hidden_activations, training=True)

            # use propabilitys instead of actual states -> faster
            # Traverse the data to match the matrix dimensions
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
            # Update weights.

            self.weights += learning_rate * (pos_associations - neg_associations) / num_examples
            if self.debug:
                error = np.sum((data - neg_visible_probs) ** 2)

        print("Run %s: error is %s" % (step_number, error) if self.debug else "Run %s" % step_number)

    def _save_weights(self, runs, dsize):
        filename = os.getcwd() + self.weights_dir + 'w' + str(runs) + 's' + str(dsize) + 'h' + str(
            self.num_hidden) + 'v' + str(
            self.num_visible)
        try:
            np.save(filename, self.weights)
        except IOError:
            print("Oops, could not save weights file to: " + filename + ".npy")

    def train(self, data, max_runs=1000, learning_rate=0.1, example_size=0):
        """
        Train the machine.
        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.
        max_runs: Number of training runs
        learning_rate: Weight of one training run
        example_size: Number of examples
        """
        # Insert bias units of 1 into the first column.
        train_data = np.insert(data, 0, 1, axis=1)
        num_examples = data.shape[0]
        bias = None
        start_time = datetime.datetime.now()
        if self.cuda:
            cm.cuda_set_device(0)
            cm.init()
            self.weights_cuda = cm.CUDAMatrix(self.weights)
            train_data = cm.CUDAMatrix(train_data)
            bias = cm.CUDAMatrix(np.asarray(ones((num_examples, 1))))
        for run in range(max_runs):
            self._one_training_step(train_data, run, learning_rate, num_examples, bias, train_data.transpose())
            if run % 100 == 0 and run != 0:
                now = datetime.datetime.now()
                total = (now - start_time).total_seconds()
                total_m = total / 60
                avg = total / run
                estimate = avg * max_runs / 60
                print("average time per round %ss, total runtime is %sm, left: %sm" % (avg, total_m, estimate))
        if self.cuda:
            self.weights = self.weights_cuda.asarray()
            cm.shutdown()
        if self.save:
            self._save_weights(max_runs, example_size)

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.

        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.weights)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._propability(hidden_activations)
        # Turn the hidden units on with their specified probabilities.
        hidden_states[:, :] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # Always fix the bias unit to 1.
        # hidden_states[:,0] = 1

        # Ignore the bias units.
        hidden_states = hidden_states[:, 1:]
        return hidden_states

    def run_hidden(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of hidden units, to get a sample of the visible units.
        Parameters
        ----------
        data: A matrix where each row consists of the states of the hidden units.
        Returns
        -------
        visible_states: A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
        """

        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the visible units (plus a bias unit)
        # sampled from a training example.
        visible_states = np.ones((num_examples, self.num_visible + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the visible units.
        visible_activations = np.dot(data, self.weights.T)
        # Calculate the probabilities of turning the visible units on.
        visible_probs = self._propability(visible_activations)
        # Turn the visible units on with their specified probabilities.
        visible_states[:, :] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        # Always fix the bias unit to 1.
        # visible_states[:,0] = 1

        # Ignore the bias units.
        visible_states = visible_states[:, 1:]
        return visible_states

    def load_weights(self, runs, dsize):
        filename = os.getcwd() + self.weights_dir + 'w' + str(runs) + 's' + str(dsize) + 'h' + str(
            self.num_hidden) + 'v' + str(
            self.num_visible) + ".npy"
        try:
            self.weights = np.load(filename)
        except IOError:
            print("Oops could not load weights file: " + filename)


def read_mnist(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)


def process_mnist_data(mnist_data, num_exmpl):
    """
    Python function for reshaping the mnist image data from a shape 28 x 28 matrix
    with a separate label to the form 794,0 vector. The 10 first spaces are the label
    neurons
    :param mnist_data: mnist ara with label and image tupels
    :param num_exmpl: size of the examples
    :return: returns an array with flat label + image vectors
    """
    processed_img = []
    for idx in xrange(0, num_exmpl):
        label, image_data = mnist_data[idx]
        pic_and_lab = pic_to_lab_vec(label, image_data)
        processed_img.append(pic_and_lab)
    return np.asarray(processed_img)


def pic_to_lab_vec(label, image_data):
    """
    Process label and a 28x28 image to a 794,0 vector
    :param label: label with the number corresponding to the picture
    :param image_data: 28x28 image
    :return: 794 label + image vector
    """
    current_label_in_neurons = [0] * 10
    current_label_int = label
    # set the right neuron to label input
    current_label_in_neurons[current_label_int] = 1
    image_new = []
    image_data = np.ravel(image_data)
    for idx1 in xrange(0, image_data.size):
        image_new.append(image_data[idx1])
    image_new[:0] = current_label_in_neurons
    fixed_image = np.asarray(image_new)
    # set all visible pixels to 1
    fixed_image[fixed_image > 0] = 1
    return fixed_image


def lab_vec_to_pic(data):
    """
    Compute the 794 image back to a label, 28x28 img tupel
    :param data: vektor with the form label + image data, size 794
    :return: label and img tupel
    """
    number = 0
    for indx in xrange(0, 10):
        if data[indx] == 1:
            number = indx
    real_image = data[10:794]
    real_image.shape = (28, 28)
    return number, real_image


# init application parameter
TRAIN_DATA_SIZE = 5000
TRAIN_RUNS = 4000
LEARNING_RATE = 0.1
# we have 28 * 28 pixels -> 784 pixels total + numbers from 0 - 9 as labels
# so we have a total input of 794 values or 794 input nodes
VISIBLE = 794
HIDDEN = 794
USE_CUDA = False

# load and process data for feeding to the rbm
trainings_data = list(read_mnist("training", "./Mnist"))
images_data = process_mnist_data(trainings_data, TRAIN_DATA_SIZE)

# init und train the rbm
rbm = RBM(num_visible=VISIBLE, num_hidden=HIDDEN, persisten=True, with_cuda=USE_CUDA)

#rbm.train(images_data, max_runs=TRAIN_RUNS, learning_rate=LEARNING_RATE, example_size=TRAIN_DATA_SIZE)

# load  saved training data
rbm.load_weights(runs=30000, dsize=2000)

test_label, test_img = trainings_data[9015]
test_img_vec = pic_to_lab_vec(test_label, test_img)

hidden_units = rbm.run_visible(np.asarray([test_img_vec]))
visible_units = rbm.run_hidden(hidden_units)

computed_lab, computed_img = lab_vec_to_pic(visible_units[0])

# draw the original input image and the rbm predicted image
plt.figure(1)
plt.imshow(test_img, interpolation='nearest')
plt.figure(2)
plt.imshow(computed_img, interpolation='nearest')
plt.show()
