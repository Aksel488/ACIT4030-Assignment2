import os
import trimesh
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import numpy as np 
import pandas as pd

def download_modelnet10():
    dataset_path = os.path.dirname(__file__)

    DATA_DIR = tf.keras.utils.get_file(
        "modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
        cache_dir=dataset_path
    )
    DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

    return DATA_DIR

# function from: https://davidstutz.de/visualizing-triangular-meshes-from-off-files-using-python-occmodel/
def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file)

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        assert lines[0] == 'OFF'

        parts = lines[1].split(' ')
        assert len(parts) == 3

        num_vertices = int(parts[0])
        assert num_vertices > 0

        num_faces = int(parts[1])
        assert num_faces > 0

        vertices = []
        for i in range(num_vertices):
            vertex = lines[2 + i].split(' ')
            vertex = [float(point) for point in vertex]
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[2 + num_vertices + i].split(' ')
            face = [int(index) for index in face]

            assert face[0] == len(face) - 1
            for index in face:
                assert index >= 0 and index < num_vertices

            assert len(face) > 1

            faces.append(face)

        return vertices, faces

def show_mesh(mesh_path, color=[], random_color=False):
    '''
    Loads a mesh file, colors and shows it

    color: RGBA array of values
    random_colors: Boolean if faces should be randomly colored
    '''
    mesh = trimesh.load(mesh_path)

    if color:
        for facet in mesh.facets:
            mesh.visual.face_colors[facet] = color

    if random_color:
        for facet in mesh.facets:
            mesh.visual.face_colors[facet] = trimesh.visual.random_color()
    
    mesh.show()
    
    
    
    
def save_plots(history, model_name):
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
    plt.savefig(f'{img_save_path}/{model_name}_accuracy.png')
    plt.clf()
    
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(f'{img_save_path}/{model_name}_loss.png')
    
def myprint(s):
    with open('classifier_summary.txt','a') as f:
        print(s, file=f)
        
        
def load_data():
    '''
    function for loading data from the training data and returning it as a Dataset
    '''
    data = np.load("modelnet10.npz", allow_pickle=True)
    train_voxel = data["train_voxel"] # Training 3D voxel samples
    test_voxel = data["test_voxel"] # Test 3D voxel samples
    train_labels = data["train_labels"] # Training labels (integers from 0 to 9)
    test_labels = data["test_labels"] # Test labels (integers from 0 to 9)
    class_map = data["class_map"] # Dictionary mapping the labels to their class names.
    
    return train_voxel, test_voxel, train_labels, test_labels, class_map        
        

        
def ensemble_scores(voxel_pred_file, pointcloud_pred_file):
    with open(voxel_pred_file) as f:
        voxel_preds = pd.DataFrame(json.load(f))
    
    with open(pointcloud_pred_file) as f:
        pointcloud_preds = pd.DataFrame(json.load(f))
        
    pointcloud_pred2 = pointcloud_preds['predictions']
    
    voxel_preds = voxel_preds.to_numpy()
    pointcloud_pred2 = pointcloud_pred2.to_numpy()
    
    final_pred = []
    for i in range(len(voxel_preds)):
        pointcloud_pred2[i] = np.array(pointcloud_pred2[i])
        final_pred.append(np.add(voxel_preds[i], pointcloud_pred2[i]) / 2)
        
    pointcloud_pred2 = np.concatenate(pointcloud_pred2)
    pointcloud_pred2 = np.reshape(pointcloud_pred2, (908, 10))
        
    return final_pred, pointcloud_pred2, voxel_preds