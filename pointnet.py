import os 
import trimesh
import numpy as np 
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, Dense, 
    GlobalMaxPooling1D, Dot, Reshape, Input, Dropout
)

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")


def save_plots(history):
    """
    Function for plotting and saving accuracy and loss of a model.
    """
    img_save_path = os.path.join('models', 'plots')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(img_save_path + '/pointnet_accuracy.png')
    plt.clf()
    
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(img_save_path + '/pointnet_loss.png')


def myprint(s):
    with open('pointnet_summary.txt','a') as f:
        print(s, file=f)
        

def parse_dataset(data_path, num_points=2048):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(data_path, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )
    
    
def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


def transform_to_tf_dataset(x_train, y_test, x_labels, y_labels, shuffle=True):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((y_test, y_labels))

    if shuffle: 
        train_dataset = train_dataset.shuffle(len(x_train)).map(augment).batch(BATCH_SIZE)
        test_dataset = test_dataset.shuffle(len(y_test)).batch(BATCH_SIZE)
        
    return train_dataset, test_dataset


def conv_bn(x, filters):
    x = Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = BatchNormalization(momentum=0.0)(x)
    return Activation("relu")(x)


def dense_bn(x, filters):
    x = Dense(filters)(x)
    x = BatchNormalization(momentum=0.0)(x)
    return Activation("relu")(x)


class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
    
def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return Dot(axes=(2, 1))([inputs, feat_T])


def create_pointnet(summary=False):
    inputs = Input(shape=(NUM_POINTS, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = Dropout(0.3)(x)

    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs, name="pointnet")
    
    if summary:
        model.summary(print_fn=myprint)
        
    return model

def main():
    if not os.path.exists('models'):
        os.makedirs('models')
        
    pointnet_model_path = os.path.join('models', 'pointnet')
    if not os.path.exists(pointnet_model_path):
        os.makedirs(pointnet_model_path)
    
    train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(DATA_DIR, NUM_POINTS)
    train_dataset, test_dataset = transform_to_tf_dataset(train_points, test_points, train_labels, test_labels)
    
    model = create_pointnet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(name='loss'),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    )

    checkpoint_path = 'models/pointnet/cp.ckpt'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,                                            
                                                    verbose=1
    )
    
    stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    callbacks = [cp_callback, stopping_callback]

    history = model.fit(
        train_dataset, 
        epochs = EPOCHS,
        callbacks=[callbacks]
    )
    
    save_plots(history)

if __name__ == "__main__":
    main()
    
    
