#!/home/jgm00105/.conda/envs/espnet/bin/python

import sys
import json
import yaml
import random
import hashlib
import pathlib
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from copy import deepcopy
from room import ExtendedRoom
from scipy.signal import fftconvolve

def example_conf_files(sim_setup_dir):

    def write_to_file(parent_dir,filename,suffix,data):
        with open(parent_dir / f'{filename}{suffix}','w') as f:
            if suffix == '.yaml':
                yaml.dump(data,f)
            elif suffix == '.json':
                json.dump(data,f,indent=4)

    channel_mapper = dict()
    instrument_layouts = dict()
    ins_names = (
        'vocals','Vocals2','drums','Bass Drum','bass',
        'Bass2','Snare','other','Crowd','Cymbal','Additional Instruments',
        'Guitar'
    )
    ins_locs = (
        [3.2,127,65],[3,130,50],[2.5,140,0],[3,130,-55],[4,119,-70],
        [3.5,122,15],[3.5,122,20],[3.5,122,-15],[3.5,122,-20],[4,118,50],[4,118,-50],
        [5,111,55]
    )
    for ins_name in ins_names:
        if ins_name in ['Snare','Crowd']:
            continue
        channel_mapper.setdefault(ins_name,len(channel_mapper))
    channel_mapper.setdefault('Snare',channel_mapper['Bass2'])
    channel_mapper.setdefault('Crowd',channel_mapper['other'])
    channel_mapper = {
        k:channel_mapper.get(k) for k
        in sorted(channel_mapper,key=lambda k: channel_mapper.get(k,100))
    }
    instrument_layouts.setdefault(
        'A',{}
    ).update(
        {ins_name:ins_loc for ins_name,ins_loc in zip(ins_names,ins_locs)}
    )
    # write the example configuration files
    write_to_file(
        sim_setup_dir,
        'channel_mapper',
        '.json',
        channel_mapper
    )
    write_to_file(
        sim_setup_dir,
        'instrument_layouts',
        '.json',
        instrument_layouts
    )
    write_to_file(
        sim_setup_dir,
        'rooms',
        '.json',
        {
            'Test Name 1':{
                'rt60':1.2,
                'room_dims':[36,21,14]
            },
            'Test Name 2':{
                'rt60':1.65,
                'room_dims':[36,21,14]
            }
        }
    )
    write_to_file(
        sim_setup_dir,
        'config',
        '.yaml',
        {
            'source_directory':'/foo/bar/source_dataset',
            'source_files_suffix':'.flac',
            'output_directory':'/foo/bar/output_dataset',
            'samplerate':44100,
            'seed':115
        }
    )

def load_config(sim_setup_dir:pathlib.Path):
    assert isinstance(sim_setup_dir,pathlib.Path)
    kwargs = dict()

    if sim_setup_dir.exists() and sim_setup_dir.is_dir():
        # Load the config args, the channel mapper and the layouts
        try:
            with open(sim_setup_dir / 'config.yaml','r') as f:
                kwargs.update(yaml.safe_load(f))
            with open(sim_setup_dir / 'channel_mapper.json','r') as f:
                kwargs.setdefault('channel_mapper',json.load(f))
            with open(sim_setup_dir / 'instrument_layouts.json','r') as f:
                kwargs.setdefault('instrument_layouts',json.load(f))
            with open(sim_setup_dir / 'rooms.json','r') as f:
                kwargs.setdefault('rooms',json.load(f))
        except FileNotFoundError:
            print(
                'Error while trying to load simulation configuration.\n'\
                'At least one of the following files is missing in the sim_setup directory:\n'\
                '\tconfig.yaml\n'\
                '\tchannel_mapper.json\n'\
                '\tinstrument_layouts.json\n'\
                '\trooms.json\n'\
                'Please, make sure to add them before re-running the simulation tool\n'\
                'Exiting now...'
            )
            sys.exit(1)
        # Load the dictionary containing the dataset filepaths
        try:
            with open(sim_setup_dir / 'dataset_files.json','r') as f:
                kwargs.setdefault('dataset_files',json.load(f))
        except FileNotFoundError:
            print(
                f'Dataset files have NOT been scanned!\n'\
                f'Scanning dataset files...'
            )
            dataset_samples = scan_dataset_directory(
                dataset_rootdir=pathlib.Path(kwargs.get('source_directory')),
                suffix=kwargs.get('source_files_suffix')
            )
            print("Hola")
            print(dataset_samples)
            kwargs.setdefault('dataset_files',dataset_samples)
            with open(sim_setup_dir / 'dataset_files.json','w') as f:
                json.dump(
                    dataset_samples,
                    f,
                    indent=4
                )
    else:
        print(
            'Simulation setup directory does NOT exist!\n'\
            'Creating directory and populating it with example files...'
        )
        sim_setup_dir.mkdir(parents=True,exist_ok=True)
        example_conf_files(sim_setup_dir)
        print(
            'Example configuration files created.\n'\
            'Please edit them before re-running the simulation tool.\n'\
            'Exiting now...'
            )
        sys.exit(1)

    return kwargs

def scan_dataset_directory(
    dataset_rootdir:pathlib.Path,
    suffix='.flac'):
    dataset_samples = dict()
    for file in dataset_rootdir.rglob(f'*{suffix}'):
        parts = file.parts
        mic = parts[-2]
        perf = parts[-3]
        dataset_samples.setdefault(perf,[]).append(str(file))
    
    return dataset_samples

def simulate_room(
    room_name,
    rt60,
    room_dims,
    fs,
    layout_name,
    instrument_layout,
    channel_mapper,
    output_rir_dir
):
    room = ExtendedRoom(
        room_name=f'{room_name}_layout_{layout_name}',
        rt60=rt60,
        room_dims=room_dims,
        fs=fs,
        instrument_layout=instrument_layout,
        channel_mapper=channel_mapper,
    )
    room_rir_filepath = output_rir_dir / room.signature / 'RIR.npy'
    if not room_rir_filepath.exists():
        print(f'Simulating {room_name}...')
        room.set_layout()
        room.save_rirs(output_rir_dir)
    else:
        print(
            f'{room_name} has been already simulated. '\
            f'Obtaining its respective RIR filepath...'
        )
    return room_rir_filepath

def dataset_split(dataset_files,rir_filepaths,seed,output_dir):
    random.seed(seed)
    # shuffle the dataset performances
    all_perfs = list(dataset_files.keys())
    random.shuffle(all_perfs)
    # initialize the dataset split results
    splits = {
        str(rir_filepath):[]
        for rir_filepath
        in rir_filepaths
    }
    # distribute in round-robin style
    for perf_idx, perf_name in enumerate(all_perfs):
        rir_idx = perf_idx % len(rir_filepaths)
        rir = str(rir_filepaths[rir_idx])
        perf = {
            perf_name:dataset_files.get(perf_name)
        }

        splits[rir].append(perf)

    return splits, seed

def format_split_metadata(
    samplerate,
    channel_mapper,
    rooms,
    instrument_layouts,
    seed,
    splits,
    output_directory
):
    # Store split metadata in a dictionary
    metadata = {
        'samplerate':samplerate,
        'channel_mapper':channel_mapper,
        'rooms':rooms,
        'instrument_layouts':instrument_layouts,
        'seed':seed,
        'splits':splits,
    }
    # Compute signature
    signature = hashlib.md5(
            json.dumps(metadata).encode()
    ).hexdigest()
    
    # write the metadata to a json file
    with open(output_directory / f'{signature}.json','w') as f:
        json.dump(metadata,f,indent=4)

    return signature,metadata

def process_performance(
    RIR_stack,
    perf_files,
    instrument_index_mapper,
    channel_mapper
):
    # IMPLEMENTATION NOT OPTIMIZED
    #  read the audio files
    dry_signals = [
        sf.read(file)[0] for file in perf_files
    ]
    # Obtain the maximum length (in samples) of the signals
    x_length = max(map(lambda x: len(x),dry_signals))
    rir_length = RIR_stack.shape[-1]
    s_length = x_length + rir_length - 1
    # Preallocate space for filtered signals
    n_mics = max(channel_mapper.values())+1
    S = np.zeros([
        n_mics,
        n_mics,
        s_length
    ])
    tqdm.write(
        f'\t* Maximum input signal length of {x_length} samples\n'\
        f'\t* RIR length of {rir_length} samples\n'\
        f'\t* Preallocated memory for array S of shape {S.shape}'
    )
    # Scale the input audio signals before filtering
    dry_signals = list(map(
        lambda x: x/abs(x).max(),
        dry_signals
    ))
    # Process audio files
    for perf_file,x in zip(perf_files,dry_signals):
        # Get the respective RIR index and mic index 
        # for this instrument
        ins_stem = pathlib.Path(perf_file).stem
        ins_idx = instrument_index_mapper.get(
            ins_stem
        )
        mic_idx = channel_mapper.get(
            ins_stem
        )
        filtered = np.apply_along_axis(
            lambda h_slice,x : fftconvolve(h_slice,x),
            axis=1,
            arr=RIR_stack[ins_idx,:,:],
            x=x
        )
        tqdm.write(
            f'Adding filtered {ins_stem} signals with shape {filtered.shape} ' \
            f'to S[{mic_idx},:,:]'
        )
        S[mic_idx,:,0:filtered.shape[-1]] = \
            S[mic_idx,:,0:filtered.shape[-1]] + filtered
    # Compute the mixture
    mixture = S.sum(axis=0)
    # ensure no clipping
    A = abs(mixture).max()
    if A > 1:
        scale_factor = .9/A
        tqdm.write(
            f'Maximum mixture value of {A:.4f}. '\
            f'Rescaling signals by {scale_factor:.4f} to prevent clipping...'
        )
        mixture = mixture * scale_factor
        S = S * scale_factor
    # obtain the reference close mic signals
    refs = S[range(n_mics),range(n_mics),:]

    return mixture, refs

def generate_dataset(
    signature,
    metadata,
    output_directory
):    
    """
    WARNING!
    The simulations are obtained by allocating the simulated signals
    in their respective rooms in a 3D array of shape
    (n_mics,n_mics,n_samples).

    The RIRs consist of 3D array of shape
    (n_instruments,n_mics,n_samples). To properly map each instrument
    with its respective RIR, the order in which they are specified in
    the instrument_layout variable is used. To ensure correct mapping,
    the same instrument order should be maintained across all instrument
    layouts.

    The microphone mixtures are computed by collapsing the resulting 
    3D array along the first dimension.
    """
    dataset_directory = output_directory / signature
    done_file = dataset_directory /'done.txt'
    pending_file = dataset_directory / 'pending.json'
    dataset_directory.mkdir(parents=True,exist_ok=True)
    # Case 1: Dataset generation already completed 
    if done_file.exists():
        print(f'{signature} dataset already completed, skipping generation.')
        return
    # Case 2: Resume dataset generation from pending.json
    elif pending_file.exists():
        print(f'Resuming {signature} dataset generation from pending file...')
        with open(pending_file, 'r') as f:
            pending_data = json.load(f)
    # Case 3: First time generation, create pending.json
    else:
        print(f'Starting new {signature} dataset generation...')
        pending_data = metadata.get('splits')
        with open(pending_file, 'w') as f:
            json.dump(pending_data, f, indent=4)
    # processing pipeline
    instrument_index_mapper = {
        k:idx for idx,k in enumerate(next(iter(
            metadata.get('instrument_layouts').values()
        )))
    }
    # Avoid modifying original pending_data while iterating
    for RIR_file, performances in tqdm(deepcopy(pending_data).items(),desc="RIRs", unit="RIR"):
        # Load the RIR from the respective file
        RIR_stack = np.load(RIR_file)
        tqdm.write(f'\nProcessing {RIR_file} assigned performances...')
        for perf in tqdm(performances, desc=f"Performances for {pathlib.Path(RIR_file).stem}", leave=False, unit="perf"):
            perf_name,perf_files = next(iter(perf.items()))
            tqdm.write(f'\nPerformance {perf_name} ({len(perf_files)} audio files)')
            # compute the mixture and obtain the reference close mic signals
            mixture, refs = process_performance(
                RIR_stack,
                perf_files,
                instrument_index_mapper,
                metadata.get('channel_mapper')
            )
            # write outputs
            output_perf_dir = dataset_directory / perf_name
            output_perf_dir.mkdir(exist_ok=True)
            tqdm.write(f'\nWriting mixture and references for {perf_name}...')
            sf.write(
                output_perf_dir / 'mixture.wav',
                mixture.T,
                metadata.get('samplerate')
            )
            sf.write(
                output_perf_dir / 'REFS.wav',
                refs.T,
                metadata.get('samplerate')
            )
            # remove processed performance from pending
            tqdm.write(f'\nRemoving {perf_name} from pending performances...')
            pending_data[RIR_file].remove(perf)
            # update pending.json atomically
            tmp_path = pending_file.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(pending_data, f, indent=4)
            tmp_path.replace(pending_file)

    # Finalize
    done_file.write_text('Dataset generation complete.')
    pending_file.unlink()   # remove the pending.json file

def main(
    source_directory,
    output_directory,
    dataset_files,
    channel_mapper,
    rooms,
    instrument_layouts,
    samplerate,
    **kwargs
):
    print(
        f'Received input arguments:\n'\
        f'\tSource directory: {source_directory}\n'\
        f'\tOutput directory: {output_directory}\n'\
        f'\tNumber of dataset samples: {len(dataset_files)}\n'\
        f'\tNumber of rooms: {len(rooms)}\n'\
        f'\tNumber of instrument layouts: {len(instrument_layouts)}\n'\
    )
    # simulate each room with every instrument layout specified
    rir_filepaths = [
        simulate_room(
            room_name=f'{room_name}_layout_{layout_name}',
            rt60=room_specs.get('rt60'),
            room_dims=room_specs.get('room_dims'),
            fs=samplerate,
            layout_name=layout_name,
            instrument_layout=instrument_layout,
            channel_mapper=channel_mapper,
            output_rir_dir=pathlib.Path(output_directory) / 'RIR'
        ) for room_name,room_specs in rooms.items()
        for layout_name,instrument_layout in instrument_layouts.items()
    ]
    # Randomly distribute the available dataset performances across the simulated rooms
    splits,seed = dataset_split(
        dataset_files,
        rir_filepaths,
        kwargs['seed'] if kwargs.get('seed') is not None else random.randint(0, 2**16),
        pathlib.Path(output_directory)
    )
    # Compute the subset signature and write metadata to json file
    signature, metadata = format_split_metadata(
        samplerate,
        channel_mapper,
        rooms,
        instrument_layouts,
        seed,
        splits,
        pathlib.Path(output_directory)
    )
    # Dataset generation pipeline
    generate_dataset(
        signature,
        metadata,
        pathlib.Path(output_directory)
    ) 

if __name__ == "__main__":
    sim_setup_dir =  pathlib.Path(
        __file__
    ).parent.resolve() / 'sim_setup'
    
    kwargs = load_config(sim_setup_dir)
    print("ei")
    
    main(**kwargs if kwargs is not None else dict())