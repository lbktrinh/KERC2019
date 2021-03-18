import os
import glob
import csv


# Folder contain entire data

path = "C:/Users/trinhle/Desktop/CODE_challenge/data"
def rearrange():

    data_file = []

    for portion in os.listdir(path):
        # portion: train, val.
        path_portion = path + "/" + portion
        for labels in os.listdir(path_portion):
            #labels: angry,disgust,fear,happy,neutral,sad,surprise
            path_labels = path_portion + "/" + labels
            for clipid in os.listdir(path_labels):
                # clipid : ID clip
                path_clip = path_labels + "/" + clipid
                print (clipid)
                clip_turn = 0
                for frames in os.listdir(path_clip):
                    # frames : list frame of clip
                    turn_no = frames.split('_')[0]
                    if int(turn_no) > clip_turn:
                        clip_turn = int(turn_no)
                for turn_id in range(1, clip_turn+1):
                    generated_files = glob.glob(os.path.join(path_clip, str(turn_id) + '*'))
                    nb_frames = len(generated_files)
                    print ('b', nb_frames)

                    data_file.append([portion, labels, clipid, str(turn_id), nb_frames])
                    print("Generated %d frames for clip %s in turn %d" % (nb_frames, clipid, turn_id))
                    print (data_file[0])
                    print (data_file[0][4])

    # In Python 2, open file with mode 'wb' instead of 'w' and remove newline
    with open(os.path.join('data','data_file.csv'), 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d clip files." % (len(data_file)))



def main():
    """

    """

    rearrange()

if __name__ == '__main__':
    main()