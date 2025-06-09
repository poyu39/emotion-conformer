import numpy as np
import soundfile as sf
import torch
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model

from dataset import PadCollator
from model import EmotionWav2vec2Conformer


def count_parameters(model: torch.nn.Module):
    frontend_params = 0
    downstream_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        if 'frontend' in name or 'wav2vec' in name:
            frontend_params += param
        else:
            downstream_params += param
        print(f'{name}: {param}')
    print(f'Frontend parameters: {frontend_params}')
    print(f'Downstream parameters: {downstream_params}')
    print(f'Total number of parameters: {frontend_params + downstream_params}')

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
    FRONTEND_MODEL_PATH = ''
    CHECKPOINT_PATH = ''
    SAMPLE_AUDIO_PATH = ''
    LABEL_MAP_PATH = ''
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    label_map : dict = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    idx_to_label = {v: k for k, v in label_map.items()}
    
    num_classes = len(label_map)
    print(f'num_classes: {num_classes}')
    
    model: Wav2Vec2Model = EmotionWav2vec2Conformer(
        checkpoint_path=FRONTEND_MODEL_PATH,
        hidden_dim=256,
        num_classes=num_classes,
        freeze_frontend=True,
        device=DEVICE
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    
    # count parameters
    count_parameters(model)
    
    model.to(DEVICE)
    model.eval()
    
    pad_collator = PadCollator()
    
    with torch.no_grad():
        print(f'SAMPLE_AUDIO_PATH: {SAMPLE_AUDIO_PATH}')
        batch = [{
            'waveform': read_audio(SAMPLE_AUDIO_PATH),
            'label': 0,
        }]
        
        input_dict = pad_collator(batch)
        
        waveform = input_dict['waveform'].to(DEVICE).float()
        padding_mask = input_dict['padding_mask'].to(DEVICE)
        
        output = model(waveform, padding_mask=padding_mask)
        print(output)
        predicted_class = torch.argmax(output, dim=1).to('cpu').numpy()[0]
        print(idx_to_label[predicted_class])
