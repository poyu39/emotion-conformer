import argparse

import numpy as np
import soundfile as sf
import torch
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from tqdm import tqdm

from dataset import IEMOCAP_Dataset, PadCollator
from model import EmotionWav2vec2Conformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frontend_model_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--label_map_path', type=str, required=True)
    
    # inference sample audio
    parser.add_argument('--audio_path', type=str, default=None)
    
    # inference dataset
    parser.add_argument('--feature_path', type=str, default=None)
    
    # export hidden states
    parser.add_argument('--export_hidden_states', type=bool, default=False)
    
    return parser.parse_args()

def read_audio(f_path):
    '''
    Read audio file and return the waveform.
    '''
    wav, sr = sf.read(f_path)
    channel = sf.info(f_path).channels
    assert sr == 16000, f'Sample rate should be 16kHz, but got {sr} in file {f_path}'
    assert channel == 1, f'Channel should be 1, but got {channel} in file {f_path}'
    
    return wav

if __name__ == '__main__':
    args = get_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    label_map : dict = np.load(args.label_map_path, allow_pickle=True).item()
    idx_to_label = {v: k for k, v in label_map.items()}
    print(f'Label map: {label_map}')
    
    model: Wav2Vec2Model = EmotionWav2vec2Conformer(
        checkpoint_path=args.frontend_model_path,
        hidden_dim=args.hidden_dim,
        num_classes=len(label_map),
        freeze_frontend=True,
        device=DEVICE
    )
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=DEVICE))
    
    model.to(DEVICE)
    model.eval()
    
    if args.audio_path is not None:
        pad_collator = PadCollator()
        with torch.no_grad():
            print(f'SAMPLE_AUDIO_PATH: {args.audio_path}')
            batch = [{
                'waveform': read_audio(args.audio_path),
                'label': 0,
            }]
            
            input_dict = pad_collator(batch)
            
            waveform = input_dict['waveform'].to(DEVICE).float()
            padding_mask = input_dict['padding_mask'].to(DEVICE)
            
            output = model(waveform, padding_mask=padding_mask)
            print(output)
            predicted_class = torch.argmax(output, dim=1).to('cpu').numpy()[0]
            print(idx_to_label[predicted_class])
    
    elif args.feature_path is not None:
        print(f'FEATURE_PATH: {args.feature_path}')
        dataset = IEMOCAP_Dataset(
            npy_path=args.feature_path,
            label_map=args.label_map_path,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            collate_fn=PadCollator(),
            shuffle=False,
        )
        
        # export hidden states
        if args.export_hidden_states:
            hidden_states = []
            hidden_states_label = []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Processing batches"):
                    input_dict = batch
                    waveform = input_dict['waveform'].to(DEVICE).float()
                    padding_mask = input_dict['padding_mask'].to(DEVICE)
                    
                    output, hidden_state = model(waveform, padding_mask=padding_mask, return_hidden_states=True)
                    hidden_states.append(hidden_state.cpu().numpy())
                    hidden_states_label.append(input_dict['label'].cpu().numpy())
            
            hidden_states = np.concatenate(hidden_states, axis=0)
            hidden_states_label = np.concatenate(hidden_states_label, axis=0)
            np.save('hidden_states_label.npy', hidden_states_label)
            np.save('hidden_states.npy', hidden_states)
            print(f'saved hidden_states.npy and hidden_states_label.npy')
