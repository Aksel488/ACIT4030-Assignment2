# ACIT4030-Final Project

# Ensemble Classifier for Different 3D Data Representations

## Setup
This project was created using the anaconda package manager. The setup commands is therefore from anaconda. Corresponding pip commands can be found by looking up the individual packages.

### Enviroment
Create a new python 3.9 environment with tensorflow gpu and activate it. If your computer does not support the use of gpu, the cpu version can be used, but note that extracting fetures from imags might be slow.
```bash
conda create -n tf-gpu python=3.9 tensorflow-gpu
conda activate tf-gpu
```
Then install the rest of the packages.
```bash
pip install numpy
pip install pyglet
pip install trimesh
pip install matplotlib
pip install pymeshlab
pip install pandas
pip install pyvista
pip install seaborn
pip install json
```

### Missing files
The folder containing the objects of ModelNet10 is not included. They will be stored in the cache once the kode has been ran.


### MeshCNN
To run the MeshCNN, you will have to speratly clone their repository. 
```bash
git clone https://github.com/ranahanocka/MeshCNN.git
cd MeshCNN
```
Running the convertToMesh.py file, will produce a seperate folder of .obj files. This folder can be copied over to the dataset folder in the MeshCNN repository.


### Running
Main code for training the individual models can be found in the files pointnet.py and voxelCNN.py. These files will generate two json files with each of their predictions.
To create the ensemble predictions, run ensemble.py. 


### folder structure
```
project
│   README.md
│   
└───ensemble.py
│    
│   
└───pointnet.py
│
│
└───setup.txt
|
|
└───utils.py
|
|
└───voxelCNN.py
```
