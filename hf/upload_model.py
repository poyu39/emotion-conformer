from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj='./model/wav2vec2-conformer-base/librispeech/checkpoints/checkpoint_best.pt',
    path_in_repo='./checkpoint.pt',
    repo_id='poyu39/wav2vec2-conformer-base_librispeech',
    repo_type='model',
)

api.upload_file(
    path_or_fileobj='./model/wav2vec2-conformer-base/wav2vec2_conformer_base_librispeech.yaml',
    path_in_repo='./wav2vec2_conformer_base_librispeech.yaml',
    repo_id='poyu39/wav2vec2-conformer-base_librispeech',
    repo_type='model',
)