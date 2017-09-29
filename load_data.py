import tensorflow as tf
import numpy as np
import os
import pandas as pd

# Extract data
def load_data():

    # Parameters
    image_dir = os.path.join('.', 'data', 'images')#os.path.join('.', 'data','small_dataset', 'images')
    num_examples = None
    image_width = 256
    image_height = 256

    # Extract y values

    # Read label column
    label_column = pd.read_csv('data/train.csv', index_col=0)#pd.read_csv('data/small_dataset/train.csv', index_col=0)

    # Make one hot vectors
    labels = pd.get_dummies(label_column['species'])
    num_examples = len(labels)
    y = np.array(labels.as_matrix(), dtype=np.float32)#tf.convert_to_tensor(labels.as_matrix())

    # Extract X values

    # Create queue of image files
    image_paths = []
    for filename in os.listdir(image_dir):
        image_paths.append(os.path.join(image_dir + "/" + filename))
    image_paths = np.asarray(image_paths)
    image_path_tensor = tf.convert_to_tensor(image_paths, dtype='string')
    image_queue = tf.train.string_input_producer(image_path_tensor, num_epochs=None)

    # Get images from files (TODO: Make more efficient)
    X = [None for i in range(num_examples)]
    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        image_reader = tf.WholeFileReader()
        for i in range(num_examples):
            key, image_file = image_reader.read(image_queue)
            image = tf.image.decode_jpeg(image_file)
            resized = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)
            X[i] = resized.eval(session=sess)
    X = np.array(X, dtype=np.float32)# X = tf.convert_to_tensor(X)

    # Make train/dev/test set

    # Split dataset
    split_value = int(num_examples * 0.8)
    X_train = X[:split_value]
    X_test = X[split_value:]
    y_train = y[:split_value]
    y_test = y[split_value:]

    return X_train, X_test, y_train, y_test
