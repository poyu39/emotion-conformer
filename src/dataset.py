import logging
from pathlib import Path

import numpy as np
import regex
import soundfile as sf


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


class DatasetPreprocessor:
    def __init__(self):
        '''
        Prepare the dataset for training.
        '''
        self.logger = logging.getLogger('DatasetPreprocessor')
        logging.basicConfig(
            format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
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
                        'wavforms': self.read_audio(wav),
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


if __name__ == '__main__':
    IEMOCAP_DATASET_PATH = '/home/poyu39/github/poyu39/emotion-conformer/dataset/IEMOCAP_full_release'
    
    OUTPUT_PATH = '/home/poyu39/github/poyu39/emotion-conformer/dataset/IEMOCAP_full_release/iemocap_dataset.npy'
    
    SELECT_EMO_LIST = [EmoType.IEMOCAP.ANGRY, EmoType.IEMOCAP.HAPPY, EmoType.IEMOCAP.SAD, EmoType.IEMOCAP.NEUTRAL]
    
    dataset_preprocessor = DatasetPreprocessor()
    buf = dataset_preprocessor.extract_iemocap_feature(IEMOCAP_DATASET_PATH, SELECT_EMO_LIST)
    dataset_preprocessor.pack_cache(buf, OUTPUT_PATH)