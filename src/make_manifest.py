import argparse
import glob
import logging
import os

import soundfile
from tqdm import tqdm

import logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'folders',
        nargs='+',
        help='One or more folders (under the same root) containing audio files',
    )
    parser.add_argument('--root', required=True, help='Common root path of all folders')
    parser.add_argument('--ext', default='flac', help='Audio file extension (default: flac)')
    parser.add_argument('--output-name', default='train', help='Output .tsv filename (train, valid, test)')
    parser.add_argument('--dest', default='.', help='Output directory')
    parser.add_argument('--dataset-name', default='dataset', help='Name of the dataset')
    return parser


def main(args):
    os.makedirs(args.dest, exist_ok=True)
    root = os.path.realpath(args.root)
    out_path = os.path.join(args.dest, f'{args.output_name}.tsv')
    
    logger = logging.getLogger('gen_manifest')
    
    logger.info('Initializing manifest generation')
    logger.info('-' * 40)
    logger.info(f'Root directory: {root}')
    logger.info(f'Input folders: {args.folders}')
    logger.info(f'Output file: {out_path}')
    logger.info('-' * 40)
    
    audio_files = []
    
    for folder in args.folders:
        folder_path = os.path.join(root, folder)
        if args.dataset_name == 'librispeech':
            search_path = os.path.join(folder_path, f'**/*.{args.ext}')
        elif args.dataset_name == 'iemocap':
            search_path = os.path.join(folder_path, f'sentences/wav/**/*.{args.ext}')
        found = glob.glob(search_path, recursive=True)
        audio_files.extend(found)
    
    audio_files = sorted(audio_files)
    
    with open(out_path, 'w') as f:
        logger.info(f'Found {len(audio_files)} audio files')
        f.write(f'{root}\n')
        for path in tqdm(audio_files, desc='Writing manifest', unit='file'):
            frames = soundfile.info(path).frames
            rel_path = os.path.relpath(path, root)
            print(f'{rel_path}\t{frames}', file=f)
    
    logger.info('Manifest generation completed successfully!')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
