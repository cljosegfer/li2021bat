
import os
import numpy as np
import h5py

from tqdm import tqdm

# from data_loader.util import *
from data_loader.util import load_label_files, load_challenge_data, resample, slide_and_cut_beat_aligned

# Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
weights_file = 'weights.csv'
normal_class = '426783006'
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

input_directory_label = '/home/josegfer/datasets/challenge2020/data'
label_dir = '/home/josegfer/datasets/challenge2020/data'
# Find the label files.
print('Finding label and output files...')
label_files = load_label_files(input_directory_label)

# # Load the labels and classes.
# print('Loading labels and outputs...')
# label_classes, labels_onehot, labels = load_labels(label_files, normal_class, equivalent_classes)

num_files = len(label_files)
# num_files = 5
# recordings2save = []
# ratio2save = []
error = []

n_lead = 12
n_segment = 10
beat_length = 400

h5f = h5py.File('data/challenge2020.h5', 'w')
X = h5f.create_dataset('recording', (num_files, n_lead, n_segment, beat_length), dtype='f8')
r = h5f.create_dataset('ratio', (num_files, 1, n_segment), dtype='f8')

for i in tqdm(range(num_files)):
    # print('{}/{}'.format(i + 1, num_files))
    recording, header, name = load_challenge_data(label_files[i], label_dir)
    recording[np.isnan(recording)] = 0

    # divide ADC_gain and resample
    recording = resample(recording, header, 500)

    # slide and cut
    try:
        recording, info2save = slide_and_cut_beat_aligned(recording, 1, 5000, 500,
                                                      seg_with_r=False, beat_length=400)
        X[i, :, :, :] = np.transpose(recording, (0, 2, 1, 3))
        r[i, :, :] = info2save
    except:
        print('skipping file: {}, idx: {}'.format(name, i))
        error.append(name)
        continue
    # print(recording)
    # print(info2save)
#     recordings2save.append(recording[0])
#     ratio2save.append(info2save)
# recordings2save = np.array(recordings2save)
# recordings2save = np.transpose(recordings2save, (0, 2, 1, 3))
# ratio2save = np.array(ratio2save)

# save_dir = 'data'

# np.save(os.path.join(save_dir, 'recordings_' + str(5000) + '_' + str(
#                 500) + '_' + str(False) + '.npy'), recordings2save)
# np.save(os.path.join(save_dir, 'info_' + str(5000) + '_' + str(
#                 500)  + '_' + str(False) + '.npy'), ratio2save)

h5f.close()
print('done')
