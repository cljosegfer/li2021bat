
import numpy as np
import h5py
import pandas as pd
import pickle

from tqdm import tqdm

from data_loader.util import slide_and_cut_beat_aligned

from scipy import signal

def just_resample(data, sample_Fs = 400, resample_Fs = 500):
    sample_len = data.shape[1]
    resample_len = int(sample_len * (resample_Fs / sample_Fs))
    resample_data = signal.resample(data, resample_len, axis=1, window=None)

    return resample_data


code15 = h5py.File('/home/josegfer/datasets/code/output/code15.h5', 'r')
metadata = pd.read_csv('/home/josegfer/datasets/code/data/exams.csv')
label_columns = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']

with open('data/remove_id', 'rb') as fp:
    remove = pickle.load(fp)
metadata_clean = metadata.copy()
for exam_id in tqdm(remove):
    metadata_clean = metadata_clean.drop(index = metadata_clean[metadata_clean['exam_id'] == exam_id].index)

indices, h5_indices, csv_indices = np.intersect1d(code15['exam_id'], metadata_clean['exam_id'], assume_unique = True, return_indices = True)

output = []
for idx, row in tqdm(metadata_clean.iterrows()):
    exam_id = row['exam_id']
    h5_idx = h5_indices[indices == exam_id]
    output.append(h5_idx[0])
metadata_clean.insert(len(metadata_clean.columns), 'h5_idx', output)
metadata_clean = metadata_clean.reset_index(drop = True)


num_files = len(metadata_clean)
n_lead = 12
n_segment = 10
beat_length = 400
n_classes = 6

h5f = h5py.File('data/code15bat.h5', 'w')

x = h5f.create_dataset('recording', (num_files, n_lead, n_segment, beat_length), dtype = code15['tracings'].dtype)
r = h5f.create_dataset('ratio', (num_files, 1, n_segment), dtype = 'f8')
y = h5f.create_dataset('label', (num_files, n_classes), dtype = 'bool')
id = h5f.create_dataset('exam_id', shape = (num_files, ), dtype = code15['exam_id'].dtype)

for idx, row in tqdm(metadata_clean.iterrows()):
    h5_idx = row['h5_idx']
    recording = code15['tracings'][h5_idx].T
    onehot = row[label_columns].to_numpy(dtype = 'bool')
    exam_id = code15['exam_id'][h5_idx]
    assert exam_id == row['exam_id']

    # resample
    recording = just_resample(recording, sample_Fs = 400, resample_Fs = 500)
    # slide and cut
    scbeat, info2save = slide_and_cut_beat_aligned(recording, n_segment = 1, window_size = 5000, sampling_rate = 500, 
                                                   seg_with_r = False, beat_length = 400)
    
    x[idx, :, :, :] = np.transpose(scbeat, (0, 2, 1, 3))
    r[idx, :, :] = info2save
    y[idx, :] = onehot
    id[idx] = exam_id

code15.close()
h5f.close()
