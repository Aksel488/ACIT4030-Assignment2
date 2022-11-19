from utils import ensemble_scores
from utils import load_data
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def custom_confusion_matrix(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    group_counts = ["{0:0.0f}".format(value) for value in conf_mat.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_mat.flatten()/np.sum(conf_mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(10, 10)
    
    ax = sns.heatmap(conf_mat, square=True, annot=labels, cmap='Blues', fmt='', cbar=False)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.tick_params(length=0, labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position('top')
    #ax.set_xticklabels(['Positive', 'Negative'])
    #ax.set_yticklabels(['Positive', 'Negative'], rotation=90, va='center')
    return ax


_, _, _, test_labels, _ = load_data()

scores, pointnet_scores, voxel_scores = ensemble_scores('voxelPred.json', 'pointPred.json')

print(accuracy_score(test_labels, np.argmax(scores, axis=-1)))
print(accuracy_score(test_labels, np.argmax(np.array(pointnet_scores), axis=-1)))
print(accuracy_score(test_labels, np.argmax(voxel_scores, axis=-1)))

plt.figure(figsize=(12,8))
custom_confusion_matrix(test_labels, np.argmax(scores, axis=-1))
plt.savefig('confusion.png', bbox_inches='tight')
plt.clf()

custom_confusion_matrix(test_labels, np.argmax(np.array(pointnet_scores), axis=-1))
plt.savefig('confusionPointNet.png', bbox_inches='tight')
plt.clf()

custom_confusion_matrix(test_labels, np.argmax(voxel_scores, axis=-1))
plt.savefig('confusionVoxel.png', bbox_inches='tight')



