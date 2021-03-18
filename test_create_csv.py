

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
    
    
    path = 'C:/Users/trinhle/Desktop/CODE_challenge'

    f = open('test_predict.csv','w',newline='')
    writer = csv.writer(f)
    writer.writerow(['id','label'])

    
    #data_type = 'images'
    data_type = 'features'
    
    N, X_test = data.get_all_sequences_in_memory_with_name('test', data_type)
    
    #X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    
   
    
    #y_test1 = np.argmax(y_test, axis=1)


    md = load_model('./data/checkpoints/weights.hdf5')
    
    optimizer = Adam(lr=1e-6)  # aggressively small learning rate
    crits = ['accuracy']
    md.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=crits)
    
    score =  md.predict(X_test)
    
    for i,a in enumerate(N):
        print (a)
        print (score[i])
        print (np.argmax(score[i]))
        
        la = np.argmax(score[i])
        
        if la == 0:
            label = 'Angry'
        if la == 1:
            label = 'Disgust'
        if la == 2:
            label = 'Fear'
        if la == 3:
            label = 'Happy'
        if la == 4:
            label = 'Neutral'
        if la == 5:
            label = 'Sad'
        if la == 6:
            label = 'Surprise'
     
            
        writer = csv.writer(f)
        writer.writerow(['{:06d}'.format(int(a))+'.mp4',label])
    f.close()
    #y_pred = np.argmax(score, axis=1)
    
    # confusion matrix
    #a = confusion_matrix(y_test1,y_pred)
    #b = a/a.sum(axis = 1, keepdims=True)
  

    
    #acc= md.evaluate(X_test,y_test, batch_size = batch_size, verbose=0)
    #print ("acc:",acc)
    
    #class_names = np.array(['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'])

if __name__ == '__main__':
    main()
