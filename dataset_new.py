# File 2: dataset.py

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def get_speaker_info(root_dir):
    speaker_ids = set()
    for domain in ['trainA', 'trainB']:
        path = os.path.join(root_dir, domain)
        if not os.path.isdir(path):
            continue
        for filepath in glob.glob(os.path.join(path, '*_spec.npy')):
            filename = os.path.basename(filepath)
            speaker_id_str = filename.split('_')[0]
            speaker_ids.add(speaker_id_str)

    if not speaker_ids:
        return 0, {}
    sorted_speakers = sorted(list(speaker_ids))
    speaker_to_id = {speaker: i for i, speaker in enumerate(sorted_speakers)}
    num_speakers = len(speaker_to_id)
    print(f"Discovered {num_speakers} unique speakers.")
    return num_speakers, speaker_to_id

class UnalignedSpectrogramDataset(Dataset):
    def __init__(self, root_dir, speaker_to_id_map, mode='train', max_len=128):
        self.root_dir = root_dir
        self.mode = mode
        self.max_len = max_len
        self.speaker_to_id = speaker_to_id_map
        self.files_A = sorted(glob.glob(os.path.join(root_dir, f'{mode}A', '*_spec.npy')))
        self.files_B = sorted(glob.glob(os.path.join(root_dir, f'{mode}B', '*_spec.npy')))
        print(f"Found {len(self.files_A)} files in {mode}A and {len(self.files_B)} files in {mode}B")

    def __getitem__(self, index):
        index_B = index % len(self.files_B)
        spec_A_path = self.files_A[index]
        spec_B_path = self.files_B[index_B]
        f0_A_path = spec_A_path.replace('_spec.npy', '_f0.npy')
        f0_B_path = spec_B_path.replace('_spec.npy', '_f0.npy')
        spec_A = torch.from_numpy(np.load(spec_A_path)).float()
        f0_A = torch.from_numpy(np.load(f0_A_path)).float()
        spec_B = torch.from_numpy(np.load(spec_B_path)).float()
        f0_B = torch.from_numpy(np.load(f0_B_path)).float()
        spec_A, f0_A = self._random_crop(spec_A, f0_A)
        spec_B, f0_B = self._random_crop(spec_B, f0_B)
        filename_A = os.path.basename(spec_A_path)
        speaker_id_str_A = filename_A.split('_')[0]
        speaker_id_A = self.speaker_to_id[speaker_id_str_A]
        filename_B = os.path.basename(spec_B_path)
        speaker_id_str_B = filename_B.split('_')[0]
        speaker_id_B = self.speaker_to_id[speaker_id_str_B]
        phon_A = torch.tensor([1.0, 0.0])
        phon_B = torch.tensor([0.0, 1.0])
        return {
            'A_spec': spec_A, 'B_spec': spec_B,
            'A_f0': f0_A, 'B_f0': f0_B,
            'phon_A': phon_A, 'phon_B': phon_B,
            'speaker_id_A': torch.tensor(speaker_id_A).long(),
            'speaker_id_B': torch.tensor(speaker_id_B).long()
        }

    def _random_crop(self, spec, f0):
        n_mels, n_frames = spec.shape
        if n_frames >= self.max_len:
            start = np.random.randint(0, n_frames - self.max_len + 1)
            spec = spec[:, start:start + self.max_len]
            f0 = f0[start:start + self.max_len]
        else:
            spec = torch.nn.functional.pad(spec, (0, self.max_len - n_frames), 'constant', 0)
            f0 = torch.nn.functional.pad(f0, (0, self.max_len - n_frames), 'constant', 0)
        return spec, f0

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))