#%% Set command line flags
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# command line flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', \
    './data/trial04/,./data/trial09/,./data/trial10/,./data/trial11/,./data/trial12/,./data/trial13/', \
    "The directory location of training data files.")
flags.DEFINE_integer('epochs', 6, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")
flags.DEFINE_float('valid_split', 0.2, "The ratio of validation dataset.")
flags.DEFINE_float('drop_rate', 0.5, "The drop rate on dropout layers.")

def print_section_header(title):
    """
    Helper function to print section header with given title
    :param title:
    :return:
    """
    print()
    print('#' * 30)
    print('#', title)
    print('#' * 30)


#%% Model hyper-parameters
EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size
VALID_SPLIT = FLAGS.valid_split
DROP_RATE = FLAGS.drop_rate
# Amount of steering measurement adjustment for the sideview images
SIDEVIEW_STEER_ADJUSTMENT = 0.2
# Number of pixels to crop top and bottom of the image
CROP_TOP, CROP_BOT = 70, 25

print_section_header('Model hyper-parameters')
print("Epochs: {}".format(EPOCHS))
print("Batch Size: {}".format(BATCH_SIZE))
print("Validation Set Split Ratio: {}".format(VALID_SPLIT))
print("Dropout Keep Probability: {}".format(DROP_RATE))
print("Sideview Steering Measure Adjustment: {}".format(SIDEVIEW_STEER_ADJUSTMENT))
print("Number of Pixels to Crop (top, bottom): {}, {}".format(CROP_TOP, CROP_BOT))


#%% Read driving log
from sklearn.model_selection import train_test_split

def read_driving_log(data_dir):
    """
    Read driving log (driving_log.csv) from given directory

    For each log line item in the file, 6 drive logs entry will be added.
    Each line item contains 3 image references.  For each, original and
    horizontally flipped images will be added.

    This function simply processes drive log files, and image files are read
    through generator for each batch.  Thus, this function simply marks
    'flip_flag' to indicate to flip an image during actual image read.

    With usage of a generator, it is important to parametrize and populate
    all image augmentation options during log processing.  In the generator,
    each image is chosen randomly (shuffle), and you don't want to pair the
    original and augmented images.

    :param data_dir:
    :return:
    """
    drive_logs = []
    log_file = data_dir + 'driving_log.csv'
    img_dir = data_dir + 'IMG/'
    with open(log_file) as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            # 6 samples are added for each line of the driving log
            img_indexes = list(range(3)) * 2
            steer_adjusters = [0.0, 1.0, -1.0] * 2
            flip_flags = [False]*3 + [True]*3
            for img_index, steer_adjuster, flip_flag in zip(img_indexes, steer_adjusters, flip_flags):
                img_filepath = img_dir + line[img_index].split('/')[-1]
                steer_measure = float(line[3]) + steer_adjuster * SIDEVIEW_STEER_ADJUSTMENT
                drive_log = [img_filepath, steer_measure, flip_flag]
                drive_logs.append(drive_log)
    return drive_logs

data_dirs = FLAGS.data_dir
data_dirs = data_dirs.split(',')
drive_logs = []
for data_dir in data_dirs:
    logs = read_driving_log(data_dir)
    drive_logs.extend(logs)

# Divide drive_logs into training and validation sets
train_logs, valid_logs = train_test_split(drive_logs, test_size=VALID_SPLIT)


#%% Print dataset summary an visualize an image
nb_total_samples = len(drive_logs)
nb_train, nb_valid = len(train_logs), len(valid_logs)
img_index_rand = np.random.randint(0, nb_total_samples)
img_file_info = drive_logs[img_index_rand]
img_rand_filepath = img_file_info[0]
img_flip = img_file_info[2]
img_rand = cv2.imread(img_rand_filepath)
img_shape = img_rand.shape

print_section_header('Dataset Summary')
print("Image Shape: {}".format(img_shape))
print("Number of Total Samples: {}".format(nb_total_samples))
print("Number of Train Samples: {}".format(nb_train))
print("Number of Validation Samples: {}".format(nb_valid))

def visualize_image(img, flip=False, img_filepath=None):
    """
    Visualize a given 'img' with cropping line overlaid.

    :param img:
    :param img_filepath:
    :return:
    """
    # Convert from BGR to RGB (also creates a copy of np.array)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw lines at the top and bottom cropping sections
    bottom = img.shape[0] - CROP_BOT
    img[CROP_TOP - 1:CROP_TOP + 1, :, 0] = 255.0
    img[bottom - 1:bottom + 1, :, 0] = 255.0

    outfile = 'image_out'
    if (img_filepath):
        plt.title(img_filepath)
        outfile = img_filepath.split('/')[-1].replace('.jpg', '')

    # Flip image when indicated
    if (flip):
        img = np.fliplr(img)
        outfile += '_flipped'
    outfile += '.jpg'

    plt.imshow(img)
    # plt.savefig(outfile)
    plt.show()

# visualize_image(img_rand, img_flip, img_rand_filepath)


#%% Batch data generator
from sklearn.utils import shuffle

def dataset_generator(drive_logs, batch_size):
    """
    A generator function that yields np.array of images and steering measurements
    from the given 'drive_logs' in 'batch_size' chunks.

    As the sample size grows, loading all samples into a memory becomes infeasible.
    Thus, the generator approach is crucial.

    :param drive_logs:
    :param batch_size:
    :return:
    """
    nb_logs = len(drive_logs)
    while (True):
        shuffle(drive_logs)
        for offset in range(0, nb_logs, batch_size):
            batch_logs = drive_logs[offset: offset + batch_size]

            # Collect dashboard images and steering measurements
            images, steer_measures = [], []
            for drive_log in batch_logs:
                # Read image file and steering measurement
                img_filepath  = drive_log[0]
                img = cv2.imread(img_filepath)
                steer_measure = drive_log[1]
                flip_flag = drive_log[2]

                # Flip the image and measurement if indicated
                if (flip_flag):
                    img = np.fliplr(img)
                    steer_measure = steer_measure * -1.0
                images.append(img)
                steer_measures.append(steer_measure)
            yield np.array(images), np.array(steer_measures)

train_generator = dataset_generator(train_logs, batch_size=BATCH_SIZE)
valid_generator = dataset_generator(valid_logs, batch_size=BATCH_SIZE)


#%% Build neural network model using Keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D, Activation, MaxPooling2D

# Image normalizer
normalizer = lambda x: (x / 255.0) - 0.5

def build_nn_model_simple():
    """
    Build a simple neural network model using Keras Sequential model.

    The model first goes through cropping and normalization layers.
    The simple model contains a single fully connected layer. (65x320x3=62400 => 1)

    :return:
    """
    # Build model structure using Keras Sequential model
    model = Sequential()
    model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOT), (0, 0)), input_shape=img_shape))
    model.add(Lambda(normalizer))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def build_nn_model_lenet():
    """
    Build a LeNet based neural network model using Keras Sequential model.

    The model first goes through cropping and normalization layers.
    It contains 2 2-D convolution layers with ReLU activations, and then
    it has 3 fully connected layers with dropout (0.5 drop rate).

    :return:
    """
    # Build model structure using Keras Sequential model
    model = Sequential()
    model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOT), (0, 0)), input_shape=img_shape))
    model.add(Lambda(normalizer))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(DROP_RATE))
    model.add(Dense(120))
    model.add(Dropout(DROP_RATE))
    model.add(Dense(84))
    model.add(Dropout(DROP_RATE))
    model.add(Dense(1))
    return model

def build_nn_model_deep():
    """
    Build a deep neural network model structure using Keras Sequential model

    The model first goes through cropping and normalization layers.
    It contains 5 2-D convolution layers with ReLU activations, and then
    it has 4 fully connected layers.

    :return:
    """
    model = Sequential()
    model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOT), (0, 0)), input_shape=img_shape))
    model.add(Lambda(normalizer))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

# Build a deep neural network model and train with ADAM optimizer
# using Mean Squared Error (MSE) as a loss function
model = build_nn_model_deep()
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch=nb_train, \
                    validation_data=valid_generator, nb_val_samples=nb_valid, \
                    nb_epoch=EPOCHS, verbose=2)
model.save('model.h5')
