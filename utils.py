import os
import trimesh
import tensorflow as tf

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