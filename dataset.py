# In dcc_vae_project/dataset.py

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def get_speaker_info(root_dir):
    """
    Scans the data directories to find all unique speakers and create an ID map.

    Args:
        root_dir (string): The root directory containing trainA and trainB folders.

    Returns:
        num_speakers (int): The total number of unique speakers.
        speaker_to_id (dict): A dictionary mapping speaker ID strings to integer indices.
    """
    speaker_ids = set()
    
    # Check both training directories to find all unique speaker IDs
    for domain in ['trainA', 'trainB']:
        path = os.path.join(root_dir, domain)
        if not os.path.isdir(path):
            print(f"Warning: Directory not found at {path}. Skipping.")
            continue
            
        for filepath in glob.glob(os.path.join(path, '*.npy')):
            filename = os.path.basename(filepath)
            # Assumes filename is 'speakerID_...'
            speaker_id_str = filename.split('_')[0]
            speaker_ids.add(speaker_id_str)

    if not speaker_ids:
        return 0, {}

    # Create a sorted list to ensure consistent mapping every time
    sorted_speakers = sorted(list(speaker_ids))
    
    # Create the mapping from speaker ID string to a unique integer (0, 1, 2, ...)
    speaker_to_id = {speaker: i for i, speaker in enumerate(sorted_speakers)}
    
    num_speakers = len(speaker_to_id)
    
    print(f"Discovered {num_speakers} unique speakers.")
    
    return num_speakers, speaker_to_id


class UnalignedSpectrogramDataset(Dataset):
    """
    A dataset class for loading unpaired spectrogram data.
    """
    # Use the version of the class that accepts the speaker map
    def __init__(self, root_dir, speaker_to_id_map, mode='train', max_len=128):
        """
        Args:
            root_dir (string): Directory with all the data.
            speaker_to_id_map (dict): The map from speaker ID string to integer.
            mode (string): 'train' or 'test'.
            max_len (int): The fixed length to crop/pad spectrograms to.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.max_len = max_len
        self.speaker_to_id = speaker_to_id_map # Store the map

        self.files_A = sorted(glob.glob(os.path.join(root_dir, f'{mode}A', '*.npy')))
        self.files_B = sorted(glob.glob(os.path.join(root_dir, f'{mode}B', '*.npy')))

        print(f"Found {len(self.files_A)} files in {mode}A and {len(self.files_B)} files in {mode}B")

    def __getitem__(self, index):
        """
        Retrieves a pair of spectrograms, one from each domain, with correct speaker IDs.
        """
        index_B = index % len(self.files_B)
        
        spec_A_path = self.files_A[index]
        spec_B_path = self.files_B[index_B]

        # Get speaker ID from filename and map it to an integer
        filename_A = os.path.basename(spec_A_path)
        speaker_id_str_A = filename_A.split('_')[0]
        speaker_id_A = self.speaker_to_id[speaker_id_str_A]

        filename_B = os.path.basename(spec_B_path)
        speaker_id_str_B = filename_B.split('_')[0]
        speaker_id_B = self.speaker_to_id[speaker_id_str_B]

        # Load the spectrograms from .npy files
        spec_A = torch.from_numpy(np.load(spec_A_path)).float()
        spec_B = torch.from_numpy(np.load(spec_B_path)).float()

        if self.mode == 'train':
            spec_A = self._random_crop(spec_A)
            spec_B = self._random_crop(spec_B)

        # Create one-hot encoded phonation labels
        phonation_A = torch.tensor([1.0, 0.0])
        phonation_B = torch.tensor([0.0, 1.0])

        return {
            'A': spec_A,
            'B': spec_B,
            'phon_A': phonation_A,
            'phon_B': phonation_B,
            'speaker_id_A': torch.tensor(speaker_id_A).long(),
            'speaker_id_B': torch.tensor(speaker_id_B).long()
        }

    def _random_crop(self, spec):
        """Crops or pads the spectrogram to a fixed length."""
        n_mels, n_frames = spec.shape

        if n_frames >= self.max_len:
            start = np.random.randint(0, n_frames - self.max_len + 1)
            spec = spec[:, start:start + self.max_len]
        else:
            padding = self.max_len - n_frames
            spec = torch.nn.functional.pad(spec, (0, padding), 'constant', 0)
        return spec

    def __len__(self):
        """Returns the length of the dataset."""
        return max(len(self.files_A), len(self.files_B)) 