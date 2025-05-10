import argparse
import glob
import os

import soundfile
from tqdm import tqdm


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
    return parser


def main(args):
    os.makedirs(args.dest, exist_ok=True)
    root = os.path.realpath(args.root)
    out_path = os.path.join(args.dest, f'{args.output_name}.tsv')
    
    print(f'📁 Root directory: {root}')
    print(f'📂 Input folders: {args.folders}')
    print(f'📝 Output file: {out_path}')
    
    audio_files = []
    
    for folder in args.folders:
        folder_path = os.path.join(root, folder)
        search_path = os.path.join(folder_path, f'**/*.{args.ext}')
        found = glob.glob(search_path, recursive=True)
        audio_files.extend(found)
    
    audio_files = sorted(audio_files)
    
    with open(out_path, 'w') as f:
        print(root, file=f)  # First line: root path
        for path in tqdm(audio_files, desc='Writing manifest', unit='file'):
            frames = soundfile.info(path).frames
            rel_path = os.path.relpath(path, root)
            print(f'{rel_path}\t{frames}', file=f)
    
    print(f'✅ Done! Manifest written to: {out_path}')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
