"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from processor import process_image
from keras.utils import to_categorical

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
    	return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self, seq_length=20, class_limit=None, image_shape=(200, 200, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = 200  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        #self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data from file."""
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[4]) >= self.seq_length and int(item[4]) <= self.max_frames and item[1] in self.classes:
                data_clean.append(item)
            #if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
			
        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        #debug1 = len(label_hot)
        #debug2 = len(self.classes)

        assert len(label_hot) == len(self.classes)

        #label_hot = label_hot[0]  # just get a single row

        return label_hot

    def split_train_val(self):
        """Split the data into train and val groups."""
        train = []
        val = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            elif item[0]== 'val':
                val.append(item)
            else:
                test.append(item)
                
        return train, val, test

    def get_all_sequences_in_memory(self, train_val, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, val, test = self.split_train_val()
        if train_val == 'train':
            data = train
        elif train_val == 'val':
            data = val
        else:
            data = test

        print("Loading %d samples into memory for %sing." % (len(data), train_val))

        X, y = [], []
        for row in data:

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    #@threadsafe_generator
    
    
    def get_all_sequences_in_memory_with_name(self, train_val, data_type):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, val, test = self.split_train_val()
        if train_val == 'train':
            data = train
        elif train_val == 'val':
            data = val
        else:
            data = test

        print("Loading %d samples into memory for %sing." % (len(data), train_val))

        a=[]
        X, y = [], []
        for row in data:

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise
            
            a.append(row[2])
            X.append(sequence)
            #y.append(self.get_class_one_hot(row[1]))
         
        return a, np.array(X)   
    
    def frame_generator(self, batch_size, train_val, data_type):
        #Return a generator that we can use to train on. There are
        #a couple different things we can return:

        #data_type: 'features', 'images'
        
        # Get the right dataset for the generator.
        train, val = self.split_train_val()
        if train_val == 'train':
            data = train
        elif train_val == 'val':
            data = val
        else:
            data = test
        #data = train if train_val == 'train' else val

        print("Creating %s generator with %d samples." % (train_val, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):

                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)
             
                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)
    

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        #path = 'data/DWIv2_ff_aug/' + sample[0] + '/' + sample[1] + '/' + sample[2] + '/'
        path = 'data/' + sample[0] + '/' + sample[1] + '/' + sample[2] + '/'
        filename = sample[3]
        images = sorted(glob.glob(path + filename + '*jpg'))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        if len(input_list) >= size:
            skip = len(input_list) // size
            # Build our new output.
            output = [input_list[i] for i in range(0, len(input_list), skip)]
            # Cut off the last one if needed.
        else:
            r = size - len(input_list)
            output=[]
            for i in range(r):
                output.append(input_list[i])
                output.append(input_list[i])
            for i in range(r,len(input_list)):
                output.append(input_list[i])
        return output[:size]


    @staticmethod
    def print_class_from_prediction(predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
