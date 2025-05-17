import logging
from pathlib import Path

import numpy as np
import regex
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import logger


class EmoType:
    '''
    Emotion type for dataset.
    '''
    class IEMOCAP:
        ANGRY = 'ang'
        HAPPY = 'hap'
        SAD = 'sad'
        NEUTRAL = 'neu'
        FRUSTRATED = 'fru'
        EXCITED = 'exc'
        FEARFUL = 'fea'
        SURPRISE = 'sur'
        DISGUST = 'dis'
        OTHER = 'xxx'
        ALL = [ANGRY, HAPPY, SAD, NEUTRAL, FRUSTRATED, EXCITED, FEARFUL, SURPRISE, DISGUST, OTHER]
        
        idx_map = {
            ANGRY: 0,
            HAPPY: 1,
            SAD: 2,
            NEUTRAL: 3,
            FRUSTRATED: 4,
            EXCITED: 5,
            FEARFUL: 6,
            SURPRISE: 7,
            DISGUST: 8,
            OTHER: 9
        }


class DatasetPreprocessor:
    def __init__(self):
        '''
        Prepare the dataset for training.
        '''
        self.logger = logging.getLogger('DatasetPreprocessor')
        self.logger.info('Initialized')
    
    def extract_iemocap_feature(self, d_path: str, emo_type: list = EmoType.IEMOCAP.ALL):
        '''
        Extract features from LibriSpeech dataset.
        Args:
            d_path: path to the dataset
            emo_type: list of emotion types to extract
        '''
        
        buf = {}
        skip = 0
        
        self.logger.info(f'Extracting features from {d_path}')
        self.logger.info(f'emo_type: {emo_type}')
        
        for index in range(1, 6):
            this_session_wav_dir = f'{d_path}/Session{index}/sentences/wav'
            
            for ses in Path(this_session_wav_dir).iterdir():
                if not ses.is_dir():
                    continue
                
                label_buffer = ''
                with Path(f'{d_path}/Session{index}/dialog/EmoEvaluation/{ses.name}.txt').open('r') as f:
                    label_buffer = f.read()
                
                for wav in ses.iterdir():
                    if not wav.is_file() or wav.suffix != '.wav':
                        skip += 1
                        continue
                    
                    # regex: Ses01F_impro01_F000	neu
                    reg = regex.search(rf'{wav.stem}\s+(\w+)', label_buffer)
                    if reg is None:
                        skip += 1
                        continue
                    wav_label = reg.group(1)
                    
                    if wav_label not in emo_type:
                        skip += 1
                        continue
                    
                    buf[wav.stem] = {
                        'waveforms': self.read_audio(wav),
                        'label': wav_label
                    }
        self.logger.info(f'Extracted {len(buf)} samples, skipped {skip} samples.')
        return buf
    
    def read_audio(self, f_path):
        '''
        Read audio file and return the waveform.
        '''
        wav, sr = sf.read(f_path)
        channel = sf.info(f_path).channels
        assert sr == 16000, f'Sample rate should be 16kHz, but got {sr} in file {f_path}'
        assert channel == 1, f'Channel should be 1, but got {channel} in file {f_path}'
        
        return wav
    
    def pack_cache(self, buf: dict, output_path: str):
        '''
        Pack the dataset into a .npy file.
        Args:
            output_path: path to the output file
        '''
        np.save(output_path, buf, allow_pickle=True)
        size = Path(output_path).stat().st_size
        self.logger.info(f'Packing {len(buf)} samples into {output_path} done')
        self.logger.info(f'File size: {size / 1024 / 1024 / 1024:.2f} GB')


class IEMOCAP_Dataset(Dataset):
    def __init__(self, npy_path: str):
        '''
        Initialize the dataset.
        Args:
            dataset_path: path to the dataset
        '''
        assert Path(npy_path).exists(), f'File {npy_path} does not exist'
        dataset_npy: np.ndarray = np.load(npy_path, allow_pickle=True)
        self.dataset: dict = dataset_npy.item()
        self.keys = list(self.dataset.keys())
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.dataset[key]
        waveform = item['waveforms']
        label = item['label']
        return {
            'waveform': waveform,
            'label': EmoType.IEMOCAP.idx_map[label]
        }


class PadCollator:
    def __init__(self, mode='train', fixed_len=None):
        assert mode in ['train', 'val', 'test'], f'Invalid mode {mode}'
        self.mode = mode
        self.fixed_len = fixed_len
        self.max_len = 0  # only meaningful in train/val
    
    def set_max_len(self, max_len):
        self.max_len = max_len
    
    def __call__(self, batch):
        waveforms = [torch.tensor(item['waveform']) for item in batch]
        labels = [item['label'] for item in batch]
        
        if self.mode == 'train':
            batch_max_len = max(w.shape[0] for w in waveforms)
            self.max_len = max(self.max_len, batch_max_len)
            pad_len = self.max_len
        elif self.mode == 'val':
            pad_len = self.max_len
        elif self.mode == 'test':
            pad_len = max(w.shape[0] for w in waveforms)
        
        # fixed_len
        pad_len = pad_len if self.fixed_len is None else self.fixed_len
        
        padded_waveforms = [torch.nn.functional.pad(w, (0, pad_len - w.shape[0])) for w in waveforms]
        waveform_tensor = torch.stack(padded_waveforms)
        label_tensor = torch.tensor(labels)
        
        return {'waveform': waveform_tensor, 'label': label_tensor}


class IEMOCAP_DataLoader:
    def __init__(
            self,
            npy_path: str,
            ratios: list = [0.8, 0.1, 0.1],
        ):
        '''
        Initialize the dataloader.
        Args:
            dataset_path: path to the dataset
            batch_size: batch size
        '''
        self.logger = logging.getLogger('IEMOCAP_DataLoader')
        self.logger.info('Initialized')
        self.logger.info(f'Loading dataset from {npy_path}')
        
        dataset = IEMOCAP_Dataset(npy_path)
        
        self.train_size = int(len(dataset) * ratios[0])
        self.val_size = int(len(dataset) * ratios[1])
        self.test_size = len(dataset) - self.train_size - self.val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [self.train_size, self.val_size, self.test_size]
        )
        self.train_dataset: Subset = train_dataset
        self.val_dataset: Subset = val_dataset
        self.test_dataset: Subset = test_dataset
    
    def get_dataloader(self, batch_size: int = 32, num_workers: int = 4, shuffle: bool = True):
        '''
        Get the dataloader.
        Args:
            batch_size: batch size
            shuffle: whether to shuffle the dataset
        '''
        
        train_collator = PadCollator(mode='train')
        val_collator = PadCollator(mode='val')
        test_collator = PadCollator(mode='test')
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=train_collator
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=val_collator
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=test_collator
        )
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    IEMOCAP_DATASET_PATH = '/home/poyu39/github/poyu39/emotion-conformer/dataset/IEMOCAP_full_release'
    
    DATASET_NPY_PATH = '/home/poyu39/github/poyu39/emotion-conformer/dataset/IEMOCAP_full_release/iemocap_dataset.npy'
    
    # SELECT_EMO_LIST = [EmoType.IEMOCAP.ANGRY, EmoType.IEMOCAP.HAPPY, EmoType.IEMOCAP.SAD, EmoType.IEMOCAP.NEUTRAL]
    
    # dataset_preprocessor = DatasetPreprocessor()
    # buf = dataset_preprocessor.extract_iemocap_feature(IEMOCAP_DATASET_PATH, SELECT_EMO_LIST)
    # dataset_preprocessor.pack_cache(buf, DATASET_NPY_PATH)
    
    iemocap_dataloader = IEMOCAP_DataLoader(DATASET_NPY_PATH)
    train_loader, val_loader, test_loader = iemocap_dataloader.get_dataloader(batch_size=32)
    for batch in train_loader:
        print(batch['waveform'].shape)
        print(batch['label'])
        break