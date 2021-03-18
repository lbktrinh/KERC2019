

import numpy as np
import csv
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from keras.optimizers import Adam
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax




def main():
    seq_length = 20
    class_limit = None
    image_shape = (80, 80, 3)
    data = DataSet(
        seq_length=seq_length,
        class_limit=class_limit,
        image_shape=image_shape
    )
    batch_size = 20
    concat = False
    
    #data_type = 'images'
    data_type = 'features'
    
    X_test, y_test = data.get_all_sequences_in_memory('val', data_type)
    y_test1 = np.argmax(y_test, axis=1)


    md = load_model('./data/checkpoints/weights.hdf5')
    
    optimizer = Adam(lr=1e-6)  # aggressively small learning rate
    crits = ['accuracy']
    md.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=crits)
    score = md.predict(X_test, batch_size=batch_size)

    y_pred = np.argmax(score, axis=1)
    
    # confusion matrix
    #a = confusion_matrix(y_test1,y_pred)
    #b = a/a.sum(axis = 1, keepdims=True)
  

    
    acc= md.evaluate(X_test,y_test, batch_size = batch_size, verbose=0)
    print ("acc:",acc)
    
    class_names = np.array(['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'])

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test1, y_pred, classes=class_names,title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test1, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')

    plt.show()

if __name__ == '__main__':
    main()
