import numpy as np
import pandas as pd
from pathlib import Path
import shutil

SEED = 42 # ensure that always the same context files get selected
GLOBAL_LENGTH = 5 # 5s is the standard for bird volcalizations
SRCS = {
    'anura': '/mnt/swap/Work/Data/Amphibians/AnuranSet/AnuranSet',
    'wabad': '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/WABAD',
    'arctic': '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/ArcticBirdSounds/DataS1/audio_annots'
    }

RATIO_WITHIN_FILE = 2
RATIO_DIFF_FILE = 4

def copy_target_and_context_files():

    for dataset in ['anura', 'wabad', 'arctic']:
        df = pd.read_csv(f'data/{dataset}_anomals.csv', index_col=0)
        paths = df.wavfilename
        paths = [Path(d) for d in paths]

        if dataset == 'arctic':
            path_with_parent = [Path(SRCS[dataset]) / f'{d.stem}{d.suffix}' for d in paths]
        else:
            path_with_parent = [Path(SRCS[dataset]) / f'{d.stem.split("_")[0]}/{d.stem}{d.suffix}' for d in paths]

        get_files_with_targe_vocalizations(dataset, path_with_parent)
        
        nr_cntxt_files = 30
        get_context_files(path_with_parent, nr_cntxt_files, dataset)


def get_files_with_targe_vocalizations(dataset, path_with_parent):
    ## GET TARGET SPECIES VOCALIZATIONS
    dest_src = Path(f'data/target_species/{dataset}_dataset')
    dest_src.mkdir(exist_ok=True, parents=True)
    for file in path_with_parent:
        dest = dest_src / file.relative_to(SRCS[dataset])
        dest.parent.mkdir(exist_ok=True, parents=True)
        if dataset == 'wabad':
            file = file.parent / f'Recordings/{file.stem}{file.suffix}'
        print('File exists: ', file, file.exists())
        shutil.copy(file, dest)
        
def get_context_files(path_with_parent, nr_cntxt_files, dataset):
    ## GET CONTEXT BY RANDOM SAMPLING
    unique_parent_dirs = np.unique([d.parent for d in path_with_parent])
    
    for parent_fold in unique_parent_dirs:
        
        np.random.seed()
        files = list(parent_fold.rglob('*.wav'))
        if not files:
            files = list(parent_fold.rglob('*.flac'))
        files.sort()
        
        sampled_ints = np.random.choice(range(1, len(files)), 
                                        size = nr_cntxt_files,
                                        replace=False)
        dest_src = Path(f'data/context/{dataset}_dataset')    
        dest_src.mkdir(exist_ok=True, parents=True) 
        
        cntxt_files = np.array(files)[sampled_ints]
        for file in cntxt_files:
            dest = dest_src / file.relative_to(SRCS[dataset])
            dest.parent.mkdir(exist_ok=True, parents=True)
            
            print('File exists: ', file, file.exists())
            shutil.copy(file, dest)
    
## Write entire dataset into one h5 file

import h5py
from tqdm import tqdm
import librosa as lb

def get_audio(file, start, end, src_dir):
    path = list(src_dir.rglob(file))[0]
    raw_audio, sr = lb.load(path)
    
    start_idx, end_idx = int(start*sr), int(end*sr)
    audio_dict = dict()
    audio_dict['before'] = raw_audio[:start_idx]
    audio_dict['after'] = raw_audio[end_idx:]
    audio_dict['during'] = raw_audio[start_idx:end_idx]
    
    audio_padded_dict = dict()
    audio_padded_dict['before'] = []
    audio_padded_dict['after'] = []
    audio_padded_dict['during'] = []
    
    for key, audio in audio_dict.items():
        if len(audio) == 0:
            continue
        nr_windows = int(np.ceil(len(audio) / sr / GLOBAL_LENGTH))
        audio_padded_dict[f"{key}"] = lb.util.fix_length(
                audio,
                size=nr_windows * GLOBAL_LENGTH * sr,
                mode='minimum',
            ).reshape(nr_windows, -1)
    return (audio_padded_dict['before'], audio_padded_dict['during'], audio_padded_dict['after']), sr

# get all target embeddings

def get_random_within_file_windows(before, after):
    combined = np.vstack([*before, *after])
    np.random.seed(SEED)
    if len(combined) >= RATIO_WITHIN_FILE:
        idxs = np.random.choice(range(len(combined)), 
                                size = RATIO_WITHIN_FILE,
                                replace=False)
    else:
        idxs = np.random.randint(len(combined), 
                                size = RATIO_WITHIN_FILE)
    return combined[idxs].tolist()

def get_target_audio(dataset, data, df, src_dir):
    for idx, event in tqdm(df.iterrows(), total=len(df)):
        data['dataset'].append(dataset)
        data['filename'].append(event.wavfilename)
        data['start'].append(event.start)
        data['end'].append(event.end)
        data['length_of_annotation'].append(event.end - event.start)
        data['label'].append(event.species)
        
        audio_tup, sr = get_audio(event.wavfilename, event.start, 
                                event.end, src_dir)
        data['sample_rate'].append(sr)
        
        before, during, after = audio_tup

        data['audio'].append(during[0].tolist())
        
        windows = get_random_within_file_windows(before, after)
        data['audio'].extend(windows)
        data['label'].extend(['within_file'] * RATIO_WITHIN_FILE)
        data['dataset'].extend([dataset] * RATIO_WITHIN_FILE)
        data['filename'].extend([event.wavfilename] * RATIO_WITHIN_FILE)
        data['sample_rate'].extend([sr] * RATIO_WITHIN_FILE)
        data['start'].extend([np.nan] * RATIO_WITHIN_FILE)
        data['end'].extend([np.nan] * RATIO_WITHIN_FILE)
        data['length_of_annotation'].extend([np.nan] * RATIO_WITHIN_FILE)
    return data

def get_context_file_audio(dataset):
    
    files = list(Path(f'data/context/{dataset}_dataset').rglob('*.wav'))
    if not files:
        files = list(Path(f'data/context/{dataset}_dataset').rglob('*.flac'))
    files.sort()
    
    all_context = dict()
    all_context['audio'] = []
    all_context['filename'] = []
    all_context['sample_rate'] = []
    
    for file in tqdm(files, total=len(files)):
        
        audio, sr = lb.load(file)
        nr_windows = int(np.ceil(len(audio) / sr / GLOBAL_LENGTH))
        audio = lb.util.fix_length(
                audio,
                size=nr_windows * GLOBAL_LENGTH * sr,
                mode='minimum',
            ).reshape(nr_windows, -1)
        audio = audio[:-1] # discard the last window, because it was probably padded
        
        all_context['audio'].extend(audio)
        all_context['filename'].extend([file.stem + file.suffix] * len(audio))
        all_context['sample_rate'].extend([sr] * len(audio))
    return all_context
        

def get_random_diff_file_windows(all_context, data, dataset):
    labels, counts = np.unique(data['label'], return_counts=True)
    nr_target_annots = sum([c for c, l in zip(counts, labels) if l not in ['within_file', 'diff_file']])
    nr_diff_file_wins = nr_target_annots * RATIO_DIFF_FILE
    np.random.seed(SEED)
    idxs = np.random.choice(range(len(all_context['audio'])), 
                            size = nr_diff_file_wins, replace=False)
    
    selected_windows = np.array(all_context['audio'])[idxs].tolist()
        
    data['audio'].extend(selected_windows)        
    data['label'].extend(['diff_file'] * nr_diff_file_wins)
    data['dataset'].extend([dataset] * nr_diff_file_wins)
    data['filename'].extend(np.array(all_context['filename'])[idxs].tolist())
    data['sample_rate'].extend(np.array(all_context['sample_rate'])[idxs].tolist())
    
    
    data['start'].extend([np.nan] * nr_diff_file_wins)
    data['end'].extend([np.nan] * nr_diff_file_wins)
    data['length_of_annotation'].extend([np.nan] * nr_diff_file_wins)
    
    return data

def write_dataset_to_file(file, data):
    file.create_dataset("audio", data=data['audio'])  # no compression

    # Store metadata as fixed-length strings for fast reads
    filenames = np.array([m.encode("utf8") for m in data['filename']])
    labels = np.array([m.encode("utf8") for m in data['label']])
    datasets = np.array([m.encode("utf8") for m in data['dataset']])
    sample_rates = np.array([m for m in data['sample_rate']])
    starts = np.array([m for m in data['start']])
    ends = np.array([m for m in data['end']])
    length_of_annotations = np.array([m for m in data['length_of_annotation']])

    file.create_dataset("filenames", data=filenames)
    file.create_dataset("labels", data=labels)
    file.create_dataset("samplerates", data=sample_rates)
    file.create_dataset("datasets", data=datasets)
    file.create_dataset("starts", data=starts)
    file.create_dataset("ends", data=ends)
    file.create_dataset("length_of_annotations", data=length_of_annotations)

    file.attrs["description"] = f"""
    Dataset of species vocalizations unknown to all models. 
    Ratio of context windows from within the same file to target windows = {RATIO_WITHIN_FILE}.
    Ratio of context windows from different files of the same dataset to target windows = {RATIO_DIFF_FILE}.
    """
        
def collect_audio_segments():
    data = {
        'dataset': [],
        'label': [],
        'audio': [],
        'filename': [],
        'sample_rate': [],
        'start': [],
        'end': [],
        'length_of_annotation': [],
    }
    
    for dataset in ['arctic', 'wabad', 'anura']:
    
        df = pd.read_csv(f'data/{dataset}_anomals.csv', index_col=0)
        src_dir = Path('data/target_species') / (dataset + '_dataset')


        get_target_audio(dataset, data, df, src_dir)
        
        all_context = get_context_file_audio(dataset)
        
        get_random_diff_file_windows(all_context, data, dataset)
    
    return data
        

def create_dataset():
    data = collect_audio_segments()
    
    file_name = f"unknown_sounds_{RATIO_WITHIN_FILE}_within_file_{RATIO_DIFF_FILE}_diff_file_5s"
    with h5py.File(f"data/{file_name}.h5", "w") as f:
        write_dataset_to_file(f, data)
        

def read_dataset():
    dataset_files = list(Path('data').rglob('*.h5'))
    
    file = dataset_files[0]
    with h5py.File(file, 'r') as f:
        t = f['target_audio']
        
    print('loaded')
    
create_dataset()