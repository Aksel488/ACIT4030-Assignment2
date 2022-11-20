import os, shutil
import pymeshlab
from utils import show_mesh, download_modelnet10

'''
pymeshlab documentation:
https://pymeshlab.readthedocs.io/en/latest/filter_list.html
'''

def download_modelnet():
    path_here = os.path.dirname(__file__)
    DATA_DIR = os.path.join(path_here, 'datasets/ModelNet10')
    if not os.path.isdir(DATA_DIR):
        DATA_DIR = download_modelnet10()

def convert_modelnet_off_obj():
    '''
    Will convert every off file from to obj file storing them in a new folder.
    Will delete every file in the save destination if the save folder exist beforehand.
    '''

    LOAD = 'datasets/ModelNet10_off'
    SAVE = 'datasets/ModelNet10_obj'

    if not os.path.exists(LOAD):
        print(f'{LOAD} not found')
        return

    if os.path.exists(SAVE):
        print(f'removing {SAVE}')

        for filename in os.listdir(SAVE):
            file_path = os.path.join(SAVE, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (SAVE, e))


    ms = pymeshlab.MeshSet()
    class_dir = os.listdir(LOAD)
    i = 1

    # loop over classes
    for dir in class_dir:
        if dir.startswith('.'):
            continue

        tt_dir = os.listdir(f'{LOAD}/{dir}')

        # loop over train tests 
        for tt in tt_dir:
            if tt.startswith('.'):
                continue

            files = os.listdir(f'{LOAD}/{dir}/{tt}')
            os.makedirs(f'{SAVE}/{dir}/{tt}')

            # loop over off files
            for file in files:
                if file.startswith('.'):
                    continue

                # load mesh from off file
                path = f'{LOAD}/{dir}/{tt}/{file}'
                ms.load_new_mesh(path)

                # remove faces with zero area
                ms.meshing_remove_null_faces()

                # save mesh
                new_path = f'{SAVE}/{dir}/{tt}/{file}'
                base_path = os.path.splitext(new_path)[0]
                ms.save_current_mesh(f'{base_path}.obj')

                print('{:04d}/4899'.format(i), end='\r')
                i += 1
    
    return

def simplify_mesh_hard(path, target):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    
    m = ms.current_mesh()
    print('Before removing zero area, mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')

    ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=target, preservenormal=True)
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_null_faces()

    m = ms.current_mesh()
    print('After removing zero area, mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')
    ms.save_current_mesh(path)

    return

def simplify_mesh_smooth(path, target):
    '''
    Simplifies mesh by first aggresivly collapsing edges, 
    then smoothly remove the last 100.
    Will override current mesh.
    Also removes zero area face and close ALL boundries
    '''

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)

    m = ms.current_mesh()
    old_num_v = m.vertex_number()
    old_num_f = m.face_number()

    TARGET = target

    numFaces = 100 + 2 * TARGET

    # remove null faces before decimation
    ms.meshing_remove_null_faces()

    #Simplify the mesh. Only first simplification will be agressive
    while (ms.current_mesh().vertex_number() > TARGET):
        # ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True) # OLD

        # we preserve as much as possible from the original mesh
        ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces, preservenormal=True) # NEW
        numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)

    # Extreamly important to have manifold meshes without zero area faces
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_null_faces()

    m = ms.current_mesh()
    new_num_v = m.vertex_number()
    new_num_f = m.face_number()

    # print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')

    ms.save_current_mesh(path)

    return old_num_v, new_num_v, old_num_f, new_num_f


def simplify_modelnet(target=10000):
    BASE = 'datasets/ModelNet10_obj'

    tot_old_num_v = 0
    tot_old_num_f = 0
    tot_new_num_v = 0
    tot_new_num_f = 0
    tot_remove_v  = 0
    tot_remove_f  = 0

    max_v = 0
    max_f = 0
    new_high_v = 0
    new_high_f = 0

    class_dir = os.listdir(BASE)
    i = 1

    # loop over classes
    for dir in class_dir:
        if dir.startswith('.'):
            continue

        tt_dir = os.listdir(f'{BASE}/{dir}')

        # loop over train tests 
        for tt in tt_dir:
            if tt.startswith('.'):
                continue

            files = os.listdir(f'{BASE}/{dir}/{tt}')

            # loop over off files
            for file in files:
                if file.startswith('.') or file == 'cache':
                    continue

                # path to mesh
                path = f'{BASE}/{dir}/{tt}/{file}'

                # if "opt_HR" not in file:
                #     os.remove(path)

                old_num_v, new_num_v, old_num_f, new_num_f = simplify_mesh_smooth(path, target)

                tot_old_num_v += old_num_v
                tot_old_num_f += old_num_f
                tot_new_num_v += new_num_v
                tot_new_num_f += new_num_f
                tot_remove_v  += old_num_v - new_num_v
                tot_remove_f  += old_num_f - new_num_f

                if old_num_v > max_v: max_v = old_num_v
                if old_num_f > max_f: max_f = old_num_f

                if new_num_v > new_high_v: new_high_v = new_num_v
                if new_num_f > new_high_f: new_high_f = new_num_f

                # # To only remove faces with zero area and close holes as is required
                # ms = pymeshlab.MeshSet()
                # ms.load_new_mesh(path)
                # ms.meshing_remove_null_faces()
                # ms.meshing_close_holes()
                # ms.save_current_mesh(path)

                print('{:04d}/4899'.format(i), file, tot_remove_v, tot_remove_f)
                i += 1
    

    # print(f'highest number of vertices: {max_v}, faces: {max_f}')
    # out:    highest number of vertices: 502603, faces: 403541

    print(f'tot_old_num_v: {tot_old_num_v}, tot_new_num_v: {tot_new_num_v}, \
            tot_old_num_f: {tot_old_num_f}, tot_new_num_f: {tot_new_num_f}, \
            tot_remove_v: {tot_remove_v}, tot_remove_f: {tot_remove_f}')

    print(f'new highest: v = {new_high_v}, f = {new_high_f}')

    # 0987/4899 10022684 10304958
    # 1913/4899 10877872 10242201
    # 4899/4899 20877562 21938767

    return




if __name__ == '__main__':
    '''
    step 1: convert off to obj
    '''
    # convert_modelnet_off_obj()

    '''
    step 2: simplify
    '''
    simplify_modelnet(target=2000)

    '''
    step 3: individuall modifications of objeckts 
    '''
    # PATH = 'datasets/shrec_16/lamp/train/T16.obj'
    # PATH = 'datasets/bathtub_0001_simplified.obj'
    # PATH = 'datasets/ModelNet10_obj/table/train/table_0057.obj'

    # simplify_mesh_hard(PATH, target=1000)

    '''
    show the object in PATH
    '''
    # show_mesh(PATH)



# @software{pymeshlab,
#   author       = {Alessandro Muntoni and Paolo Cignoni},
#   title        = {{PyMeshLab}},
#   month        = jan,
#   year         = 2021,
#   publisher    = {Zenodo},
#   doi          = {10.5281/zenodo.4438750}
# }