import argparse

import fairseq.checkpoint_utils
import soundfile as sf
import torch
from fairseq.models.wav2vec.wav2vec2 import \
    Wav2Vec2Model as Wav2Vec2ConformerModel


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
    '''
    This script is used to test the Wav2Vec2ConformerModel.
    Args:
        --model_path: path to the pretrained model
        --audio_path: path to the audio file
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--audio_path', type=str)
    args = parser.parse_args()
    
    PRETRAINED_MODEL_PATH = args.model_path
    SAMPLE_AUDIO_PATH = args.audio_path
    
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([PRETRAINED_MODEL_PATH])
    model: Wav2Vec2ConformerModel = model[0]
    model.to('cuda')
    model.eval()
    print(model)
    
    x = read_audio(SAMPLE_AUDIO_PATH)
    print(x.shape)
    print(x)
    
    with torch.no_grad():
        source = torch.from_numpy(x).unsqueeze(0).to('cuda').float()
        source = source.view(1, -1)
        features = model.extract_features(source, padding_mask=None)
        x : torch.Tensor = features['x']
        output = x.squeeze(0).to('cpu').float()
        
        print(output)
        print(output.shape)