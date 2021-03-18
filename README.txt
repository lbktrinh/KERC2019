
***Enviroment Setup:

-CUDA version  V9.0.176
-Cudnn version v7.1
-Anaconda Python 3.6.8
-Tensorflow 1.13.1
-Keras 2.2.4
-Library in the requirement.txt file (if needed)

*** Step to Step:

-Step 1: get_mainface_frame.py : run file to extract the main face in each frame of the video clip. You need to put the right path of the data folder. For example, the folder "data_example/data_try" will create the folder "data_example/data_try_out"
-Step 2: data_rearrange.py: run file to create the "data_file.csv" to save the information of data. You need to put the right path of the data folder of Step 1 result. For detail, we copy the sub folder train and val from folder "data_try_out" to folder "data" and then run file data_rearrange.py 
-Step 3: extract_features.py: run file to extract the feature of the data by VGG16 pretrain on VGGFace dataset. (you can change the other pretrain model in the extractor.py file). Running this file will create sub folder "sequences" in folder "data".
-Step 4: train.py: run file to training the baseline model (you can create new model in the models.py file)
-Step 5: test.py: run file to testing, it will calculate the accuracy and draw confusion matrix.
-step 6: test_create_csv.py: run file to create csv file for submission the kaggle. We need to prepare the file like folder "data_model_test_kaggle" and rename it into "data" before we run test_create_csv.py