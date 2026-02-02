import numpy as np
import pandas as pd
from pathlib import Path
import shutil

SEED = 42 # ensure that always the same context files get selected
GLOBAL_LENGTH = 3 # 5s is the standard for bird volcalizations
src_path = '/media/siriussound/Extreme SSD/Recordings'
SRCS = {
    'anura': f'{src_path}/Amphibians/AnuranSet/AnuranSet',
    'wabad': f'{src_path}/terrestrial/Birds/WABAD',
    'arctic': f'{src_path}/terrestrial/Birds/ArcticBirdSounds/DataS1/audio_annots'
    }

RATIO_WITHIN_FILE = 2
RATIO_DIFF_FILE = 1
PAD_FUNC = 'wrap'

# number of context files to copy to the get the contextual segments from
NR_CNTXT_FILES = 50

np.random.seed(SEED)


def copy_target_and_context_files():

    for dataset in ['anura', 'wabad', 'arctic']:
        df = pd.read_csv(f'data/{dataset}_anomals.csv', index_col=0)
        paths = df.wavfilename
        paths = [Path(d) for d in paths]

        if dataset == 'arctic':
            path_with_parent = [Path(SRCS[dataset]) / f'{d.stem}{d.suffix}' for d in paths]
        else:
            path_with_parent = [Path(SRCS[dataset]) / f'{d.stem.split("_")[0]}/{d.stem}{d.suffix}' for d in paths]

        get_files_with_target_vocalizations(dataset, path_with_parent)
        
        get_context_files(path_with_parent, NR_CNTXT_FILES, dataset)


def get_files_with_target_vocalizations(dataset, path_with_parent):
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
    if dataset == 'wabad':
        unique_parent_dirs = np.unique([d.parent.parent for d in path_with_parent])
    else:
        unique_parent_dirs = np.unique([d.parent for d in path_with_parent])
    
    for parent_fold in unique_parent_dirs:
        
        files = list(parent_fold.rglob('*.wav'))
        if not files:
            files = list(parent_fold.rglob('*.flac'))
        files.sort()
        
        # exclude files that contain target vocalizations
        target_files = [f.stem for f in Path(f'data/target_species/{dataset}_dataset').rglob('*.*')]
        no_target_files = [f for f in files if not f.stem in target_files]
        
        sampled_ints = np.random.choice(range(1, len(no_target_files)), 
                                        size = nr_cntxt_files,
                                        replace=False)
        dest_src = Path(f'data/context/{dataset}_dataset')    
        dest_src.mkdir(exist_ok=True, parents=True) 
        
        cntxt_files = np.array(no_target_files)[sampled_ints]
        for file in cntxt_files:
            dest = dest_src / file.relative_to(SRCS[dataset])
            dest.parent.mkdir(exist_ok=True, parents=True)
            
            print('File exists: ', file, file.exists())
            shutil.copy(file, dest)
    
## Write entire dataset into one h5 file

import h5py
from tqdm import tqdm
import librosa as lb


def get_audio(file, target_start, target_end, src_dir, other_target_segments):
    path = list(src_dir.rglob(file))[0]
    raw_audio, sr = lb.load(path)
    
    # get all the start and end positions of target segments
    segment_indices = [(target_start, target_end), *other_target_segments]
    segment_indices.sort()
    segments = []
    
    # extract the segments from the raw audio corresponding to
    # before, during and after the target segment
    # as well as segments in between annotated segments
    # and other annotated segments
    last_end = 0
    for idx, (start, end) in enumerate(segment_indices):
        if start == target_start:
            target_idx = idx
        start, end = int(start*sr), int(end*sr)
        # noise segments before annotations are in even-numbered indices
        segments.append(raw_audio[last_end:start])
        # annotated segments are in odd-numbered indices
        segments.append(raw_audio[start:end])
        last_end = end
    
    # create a dictionary where we store the segments and their corresponding starting times
    audio_dict = dict()
    audio_dict['during'] = {'audio': segments[2*target_idx + 1], 'start': target_start}
    audio_dict['before'] = {'audio': segments[0], 'start': 0}
    
    # these are the segments inbetween annotated segments
    # we're only collecting the even-numbered indices, hence 
    # we are only collecting noise
    for i in range(int(len(segments)/2)):
        if i > 0 and i != target_idx and len(segments[2*i]) > sr:
            audio_dict[i] = {
                'audio': segments[2*i],
                # the noise starts with the end of the previous annotated timestamp
                'start': segment_indices[i-1][1]
                }
            
    # finally we collect the noise at the end of the file
    audio_dict['after'] = {'audio': raw_audio[last_end:], 'start': last_end/sr}
    
    # the audio_dict now contains one annotated segment and the rest is noise
    # because we only took one odd-numbered index from segments, that's the
    # current target annotation based on the annotation from the dataframe 
    # that was passed to this file, the rest are the even-numbered indices
    # of the segments array, which are all noise in between annotations
    
    audio_padded_dict = dict()
    starts, ends = [], []
    
    # now create a dictionary where we create padded segments each corresponding to the
    # GLOBAL_LENGTH and named the same way as the previous dictionary
    for key, d in audio_dict.items():
        audio = d['audio']
        if len(audio) == 0:
            continue
        nr_windows = int(np.ceil(len(audio) / sr / GLOBAL_LENGTH))
        audio_padded_dict[key] = lb.util.fix_length(
                audio,
                size=nr_windows * GLOBAL_LENGTH * sr,
                mode=PAD_FUNC,
            ).reshape(nr_windows, -1)
        if False:#Sanity check
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1,audio_padded_dict[key].shape[0])
            for idx, ax in enumerate(axes):
                S = lb.feature.melspectrogram(
                    y=audio_padded_dict[key][idx],
                    sr=sr,
                    fmax=8000
                    )
                S_dB = lb.power_to_db(S)
                img = lb.display.specshow(S_dB, ax=ax)
            fig.savefig('sanity_check.png')
                
        if not key == 'during':
            starts.extend(np.arange(nr_windows) * GLOBAL_LENGTH + d['start'])
            ends.extend(np.arange(1, nr_windows) * GLOBAL_LENGTH + d['start'])
            
            # For the last ends entry:
            # take the last starts entry and add the modulo of
            # the length of audio in seconds by the global length
            # this should correspond to the fragment length in seconds
            ends.append(starts[-1] + len(audio)/sr % GLOBAL_LENGTH)
    starts.sort()
    ends.sort()
    
    # create a context_audio list that contains all context audio segments
    context_audio = []
    if 'before' in audio_padded_dict:
        context_audio.extend(audio_padded_dict['before'])
    for k, aud in audio_padded_dict.items():
        if isinstance(k, int):
            context_audio.extend(aud)
    if 'after' in audio_padded_dict:
        context_audio.extend(audio_padded_dict['after'])
    # ensure the number of starting points and the context audio segments line up
    assert len(context_audio) == len(starts) == len(ends)
    return (audio_padded_dict['during'], context_audio, starts, ends), sr

# get all target embeddings

def remove_segments_that_are_already_context(starts, ends, data, combined):
    prev_starts = np.array(data['start'])[
        np.array(data['filename'])==np.array(data['filename'][-1])
        ]
    remove = []
    for i in range(len(combined)):
        if starts[i] in prev_starts:
            remove.append((starts[i], ends[i]))
    for start, end in remove:
        idx = np.where(starts==start)[0][0]
        starts.remove(start)
        ends.remove(end)
        combined = np.vstack([combined[:idx], combined[idx+1:]])
    return combined
    

def get_random_within_file_windows(context, starts, ends, data):
    if len(context) > 0:
        combined = np.vstack(context)
    else:
        return []
    
    combined = remove_segments_that_are_already_context(starts, ends, data, combined)

            
    if len(combined) >= RATIO_WITHIN_FILE:
        idxs = np.random.choice(range(len(combined)), 
                                size = RATIO_WITHIN_FILE,
                                replace=False)
    elif len(combined) == 0:
        return []
    else:
        idxs = list(range(len(combined)))
    # write the index of the embedding to data['embed_idx']
    for idx in idxs:
        data['start'].append(starts[idx])
        data['end'].append(ends[idx])
    
    return combined[idxs]

def get_other_target_segments_from_same_file(df, current_idx):
    starts, ends = [], []
    for idx, event in df.iterrows():
        if idx == current_idx:
            continue
        else:
            starts.append(event.start)
            ends.append(event.end)
    return list(zip(starts, ends))
        

def get_target_audio(dataset, data, df, src_dir):
    # df = df[df.species == 'Black-bellied Plover']
    for idx, event in tqdm(df.iterrows(), 
                           total=len(df), 
                           desc=f'get {dataset} target audio'):
        # get other events in same file with different timestamps
        other_target_segments = []
        if len(df[df['wavfilename']==event.wavfilename]) > 1:
            df_with_same_filename = df[df['wavfilename']==event.wavfilename]
            other_target_segments = get_other_target_segments_from_same_file(df_with_same_filename, idx)
            
        audio_tup, sr = get_audio(
            event.wavfilename, 
            event.start, 
            event.end, 
            src_dir,
            other_target_segments
            )
        
        during, context, starts, ends = audio_tup

        if len(data['audio']) == 0:
            data['audio'] = during
        else:
            data['audio'] = np.vstack([data['audio'], during])
        
        data['sample_rate'].extend([sr] * len(during))
        data['dataset'].extend([dataset] * len(during))
        data['filename'].extend([event.wavfilename] * len(during))
        data['start'].extend([event.start] * len(during))
        data['end'].extend([event.end] * len(during))
        data['length_of_annotation'].extend([event.end - event.start] * len(during))
        data['label'].extend([event.species] * len(during))
        
        windows = get_random_within_file_windows(context, starts, ends, data)
        if len(windows) > 0:
            data['audio'] = np.vstack([data['audio'], windows])
        
        data['label'].extend(['within_file'] * len(windows))
        data['dataset'].extend([dataset] * len(windows))
        data['filename'].extend([event.wavfilename] * len(windows))
        data['sample_rate'].extend([sr] * len(windows))
        data['length_of_annotation'].extend([np.nan] * len(windows))
    return data

def get_context_file_audio(dataset):
    
    files = list(Path(f'data/context/{dataset}_dataset').rglob('*.wav'))
    if not files:
        files = list(Path(f'data/context/{dataset}_dataset').rglob('*.flac'))
    files.sort()
    
    all_context = dict()
    all_context['audio'] = np.array([])
    all_context['filename'] = []
    all_context['sample_rate'] = []
    
    for file in tqdm(files, total=len(files)):
        
        audio, sr = lb.load(file)
        nr_windows = int(np.ceil(len(audio) / sr / GLOBAL_LENGTH))
        audio = lb.util.fix_length(
                audio,
                size=nr_windows * GLOBAL_LENGTH * sr,
                mode=PAD_FUNC,
            ).reshape(nr_windows, -1)
        audio = audio[:-1] # discard the last window, because it was probably padded
        
        if len(all_context['audio']) == 0:
            all_context['audio'] = audio
        else:
            all_context['audio'] = np.vstack([all_context['audio'], audio])
            
        all_context['filename'].extend([file.stem + file.suffix] * len(audio))
        all_context['sample_rate'].extend([sr] * len(audio))
    return all_context

def get_embed_idxs_relative_to_files(idxs, all_context):
    # get number of segments from each audio file
    _, idx, cnts = np.unique(all_context['filename'], return_index=True, return_counts=True)
    unique_audio_files = np.array(all_context['filename'])[np.sort(idx)]
    unique_audio_file_counts = np.array(cnts)[np.argsort(idx)]
    filename_counts = {k:v for k,v in zip(unique_audio_files, unique_audio_file_counts)}
    
    # create a dictionary with cumulative counts of segments from each file
    cumulative = [0]
    cumulative_filename_counts = {}
    for i, (k, v) in enumerate(filename_counts.items(), start=1):
        cumulative.append(cumulative[i-1] + v)
        cumulative_filename_counts[k] = cumulative[i]
    
    # find the cumulative count corresponding to each index and write it into
    # a new dictionary of relative embedding index
    rel_embed_idx = []
    for idx in idxs:
        # find zero crossing, i.e. where the cumulative counts first exceed the index
        zero_crossing = np.where((idx - list(cumulative_filename_counts.values()))<0)[0][0]
        if idx < list(cumulative_filename_counts.values())[0]:
            # if the index is lower than the first entry, this means it's from the first
            # audio file
            rel_embed_idx.append(idx)
        else:
            # if the index is larger than the first entry, write the difference between 
            # the index and the next smaller cumulative count into the dictionary,
            # this will correspond to the segment index relative to the file it's from.
            rel_embed_idx.append(
                idx - list(cumulative_filename_counts.values())[zero_crossing-1]
                )
    return rel_embed_idx
        

def get_random_diff_file_windows(all_context, data, dataset):
    labels, counts = np.unique(data['label'], return_counts=True)
    # get the number of all target segments
    nr_target_annots = sum(
        [c for c, l in zip(counts, labels) 
         if l not in ['within_file', 'diff_file']]
        )
    # use number of target segemnts to calculate number of context segments
    # from different files
    nr_diff_file_wins = nr_target_annots * RATIO_DIFF_FILE
    idxs = np.random.choice(range(len(all_context['audio'])), 
                            size = nr_diff_file_wins, replace=False)
    idxs = np.sort(idxs)
    
    selected_windows = np.array(all_context['audio'])[idxs]
    
    # These don't change for any of the segments
    data['label'].extend(['diff_file'] * nr_diff_file_wins)
    data['dataset'].extend([dataset] * nr_diff_file_wins)
        
    # Nan for all noise segments
    data['length_of_annotation'].extend([np.nan] * nr_diff_file_wins)
    
    
    # add the audio segments
    data['audio'] = np.vstack([data['audio'], selected_windows])        
    
    # get filenames and sample rates based on the random indices
    data['filename'].extend(np.array(all_context['filename'])[idxs].tolist())
    data['sample_rate'].extend(np.array(all_context['sample_rate'])[idxs].tolist())
    
    # get the indices of each context segment relative to the file they originate from
    embed_idxs = get_embed_idxs_relative_to_files(idxs, all_context)
    data['start'].extend((np.array(embed_idxs) * GLOBAL_LENGTH).tolist())
    data['end'].extend(((np.array(embed_idxs)+1) * GLOBAL_LENGTH).tolist())
    
    return data

def write_dataset_to_file(file, data, chunk_size=500):
    # --- Audio ---
    # audio_data = np.array(data['audio'])
    n_samples = len(data['audio'])
    shape = data['audio'].shape
    dtype = data['audio'].dtype

    dset = file.create_dataset(
        "audio",
        shape=shape,
        dtype=dtype,
        chunks=(min(chunk_size, n_samples),) + shape[1:]
    )

    for i in range(0, n_samples, chunk_size):
        dset[i:i+chunk_size] = data['audio'][i:i+chunk_size]

    # --- Metadata ---
    dt = h5py.string_dtype(encoding="utf-8")
    file.create_dataset("filenames", data=data['filename'], dtype=dt)
    file.create_dataset("labels", data=data['label'], dtype=dt)
    file.create_dataset("datasets", data=data['dataset'], dtype=dt)
    file.create_dataset("sample_rates", data=data['sample_rate'])
    file.create_dataset("starts", data=data['start'])
    file.create_dataset("ends", data=data['end'])
    file.create_dataset("length_of_annotations", data=data['length_of_annotation'])

    file.attrs["description"] = f"""
    Dataset of species vocalizations unknown to all models. 
    Length of all sound segments = {GLOBAL_LENGTH}. 
    Ratio of context windows from within the same file to target windows = {RATIO_WITHIN_FILE}. 
    Ratio of context windows from different files of the same dataset to target windows = {RATIO_DIFF_FILE}. 
    """
        
def collect_audio_segments():
    data = {
        'dataset': [],
        'label': [],
        'audio': np.array([]),
        'filename': [],
        'sample_rate': [],
        'start': [],
        'end': [],
        'length_of_annotation': [],
    }
    
    for dataset in ['arctic', 'wabad', 'anura']:#['arctic', 'wabad', 'anura']
    
        df = pd.read_csv(f'data/{dataset}_anomals.csv', index_col=0)
        src_dir = Path('data/target_species') / (dataset + '_dataset')
        df = df.sort_values('wavfilename')

        get_target_audio(dataset, data, df, src_dir)
        
        all_context = get_context_file_audio(dataset)
        
        get_random_diff_file_windows(all_context, data, dataset)
    
    return data
        

def create_dataset():
    data = collect_audio_segments()
    
    file_name = f"unknown_sounds_{RATIO_WITHIN_FILE}_within_{RATIO_DIFF_FILE}_diff_{GLOBAL_LENGTH}s_pad_{PAD_FUNC}"
    with h5py.File(f"data/{file_name}.h5", "w") as f:
        write_dataset_to_file(f, data)
        

def read_dataset():
    dataset_files = list(Path('data').rglob('*.h5'))
    
    file = dataset_files[0]
    with h5py.File(file, 'r') as f:
        t = f['target_audio']
        
    print('loaded')
    
copy_target_and_context_files()
create_dataset()