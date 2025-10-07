import numpy as np
import pandas as pd
from pathlib import Path
import shutil


srcs = {
    'anura': '/mnt/swap/Work/Data/Amphibians/AnuranSet/AnuranSet',
    'wabad': '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/WABAD',
    'arctic': '/media/siriussound/Extreme SSD/Recordings/terrestrial/Birds/ArcticBirdSounds/DataS1/audio_annots'
    }

for dataset in ['anura', 'wabad', 'arctic']:
    df = pd.read_csv(f'{dataset}_anomals.csv', index_col=0)
    paths = df.wavfilename
    paths = [Path(d) for d in paths]

    if dataset == 'arctic':
        path_with_dad = [Path(srcs[dataset]) / f'{d.stem}{d.suffix}' for d in paths]
    else:
        path_with_dad = [Path(srcs[dataset]) / f'{d.stem.split("_")[0]}/{d.stem}{d.suffix}' for d in paths]

    ## GET TARGET SPECIES VOCALIZATIONS
    dest_src = Path(f'target_species/{dataset}_dataset')
    dest_src.mkdir(exist_ok=True, parents=True)
    for file in path_with_dad:
        dest = dest_src / file.relative_to(srcs[dataset])
        dest.parent.mkdir(exist_ok=True, parents=True)
        if dataset == 'wabad':
            file = file.parent / f'Recordings/{file.stem}{file.suffix}'
        print('File exists: ', file, file.exists())
        shutil.copy(file, dest)
    
    ## GET CONTEXT BY RANDOM SAMPLING
    
    unique_dad_folds = np.unique([d.parent for d in path_with_dad])
    nr_cntxt_files = 30
    
    for dad_fold in unique_dad_folds:
        files = list(dad_fold.iterdir())
        
        sampled_ints = np.random.randint(len(files), size = nr_cntxt_files)
        dest_src = Path(f'context/{dataset}_dataset')    
        dest_src.mkdir(exist_ok=True, parents=True) 
        
        cntxt_files = np.array(files)[sampled_ints]
        for file in cntxt_files:
            dest = dest_src / file.relative_to(srcs[dataset])
            dest.parent.mkdir(exist_ok=True, parents=True)
            
            print('File exists: ', file, file.exists())
            shutil.copy(file, dest)