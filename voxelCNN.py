import time
import os
import glob
import trimesh
import numpy as np 
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, MaxPooling3D, Concatenate, Flatten,
    BatchNormalization, InputLayer, LeakyReLU, 
    Activation, Conv3D, Reshape,
)
from utils import save_plots, myprint, download_modelnet10



DATA_DIR = download_modelnet10()
LEARNING_RATE_CLASSIFIER = 1e-3
BATCH_SIZE = 32
EPOCHS = 10
N_CLASSES = 10
    
    
def parse_dataset(data_path):
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
            train_points.append(trimesh.load(f).voxelized(3))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).voxelized(3))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )
    

def make_3Dclassifier_model():
    '''
    Create the discriminator model with structure:
    64x64x64 -> 64x32x32x32 -> 128x16x16x16 -> 256x8x8x8 -> 512x4x4x4 -> 1
    '''
    model = Sequential()

    model.add(InputLayer(input_shape=(64, 64, 64)))
    model.add(Reshape((64, 64, 64, 1)))

    model.add(Conv3D(64, (4, 4, 4), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(128, (4, 4, 4), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(256, (4, 4, 4), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv3D(512, (4, 4, 4), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.add(Reshape((32768,)))
    model.add(Dense(1))
    return model


def convert_sequential_model(classifier):
    '''
    Function code from: [https://stackoverflow.com/questions/61130836/convert-functional-model-to-sequential-keras]
    Converts Sequential model to Functional model.
    '''
    input_layer = Input(batch_shape=classifier.layers[0].input_shape)
    prev_layer = input_layer
    for layer in classifier.layers:
        layer._inbound_nodes = []
        prev_layer = layer(prev_layer)
        
    return Model([input_layer], [prev_layer]), input_layer


def voxel_classifier():
    classifier = make_3Dclassifier_model()
    classifier.build()
    classifier.pop() # Drop last Dense layer
    classifier.pop() # Drop last Reshape layer
    
    func_model, input_layer = convert_sequential_model(classifier)
    
    c2 = MaxPooling3D(pool_size=(8, 8, 8), padding='same')(func_model.layers[7].output)
    c3 = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(func_model.layers[10].output)
    c4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(func_model.layers[13].output)
    concat = Concatenate(axis=-1)([c2, c3, c4])
    out = Flatten()(concat)
    out = Dense(N_CLASSES, activation='softmax')(out)
    
    return Model(input_layer, out)


def main():
    if not os.path.exists('models'):
        os.makedirs('models')
        
    voxel_model_path = os.path.join('models', 'voxel')
    if not os.path.exists(voxel_model_path):
        os.makedirs(voxel_model_path)
    
    classifier = voxel_classifier()
    #classifier.summary(print_fn=myprint)
    
    train_voxel, test_voxel, train_labels, test_labels, class_map = parse_dataset(DATA_DIR)
    
    classifier.compile(
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_CLASSIFIER),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(name='loss'), 
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    )
    
    checkpoint_path = 'models/voxel/cp.ckpt'
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    callbacks = [cp_callback, stopping_callback]
    
    # Train the model on the training data and training labels 
    history = classifier.fit(
        train_voxel, 
        train_labels,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        shuffle = True,
        verbose = 1,
        callbacks=[callbacks]
    )
    
    save_plots(history) # Save loss and accuracy plots 

if __name__ == "__main__":
    main()