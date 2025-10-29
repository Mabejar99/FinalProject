import json
import torch
import pathlib
import soundfile as sf
from loadtensor import load_audio_chunk

class DataSample:
    def __init__(
        self,
        mix_filepath: pathlib.Path,
        ref_filepath: pathlib.Path,
        chunk_idx: int,
        start: int,
        stop: int,
        power: float = None,
    ):
        self.mix_filepath = mix_filepath
        self.ref_filepath = ref_filepath
        self.chunk_idx = chunk_idx
        self.start = start
        self.stop = stop
        self.power = self.compute_power() if power is None else power
    
    def __repr__(self):
        return ''.join([
            'DataSample(',
            f'mix_filepath={self.mix_filepath},',
            f'ref_filepath={self.ref_filepath},',
            f'chunk_idx={self.chunk_idx},',
            f'start={self.start},',
            f'stop={self.stop},',
            f'power={self.power}',
            ')'
        ])
    
    def load_audio(self,mode='both'):
        # reserva espacio para leer la señal de audio
        info = sf.info(self.ref_filepath)
        chunk_length = self.stop - self.start # sounfile lee hasta la muestra `stop` - 1
        channels = info.channels
        def load_helper(filepath,tensor):
            load_audio_chunk(
                filepath,
                self.start,
                self.stop,
                out=tensor,
            )
        if mode == 'mix':
            mix = torch.zeros([chunk_length,channels])
            # lee la señal de mezcla
            load_helper(self.mix_filepath,mix)
            return mix
        elif mode == 'refs':
            refs = torch.zeros([chunk_length,channels])
            # lee la señal de referencia
            load_helper(self.ref_filepath,refs)
            return refs
        elif mode == 'both':
            mix = torch.zeros([chunk_length,channels])
            refs = torch.zeros([chunk_length,channels])
            # lee la señal de mezcla y referencia
            for filepath,tensor in zip([self.mix_filepath,self.ref_filepath],[mix,refs]):
                load_helper(filepath,tensor)
            return mix, refs
        else:
            raise ValueError(f'Valid modes for loading audio include "mix","refs" and "both", but "{mode}" was specified')

    def compute_power(self):
        refs = self.load_audio(mode='refs')
        chunk_length,_ = refs.shape
        close_mic_mix = refs.sum(axis=1)
        power = close_mic_mix@close_mic_mix/chunk_length
        
        return power.item()

    def to_dict(self):
        return {
            'mix_filepath': str(self.mix_filepath),
            'ref_filepath': str(self.ref_filepath),
            'chunk_idx': self.chunk_idx,
            'start': self.start,
            'stop': self.stop,
            'power': self.power
        }

class MultiAcousticsSynthSOD(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: pathlib.Path,
        chunk_duration: float,
        silence_threshold: float,
        stride_duration: float,
        metadata_dir: pathlib.Path,
    ):
        self.root_dir = root_dir
        self.dataset_name = root_dir.name
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.stride_duration = stride_duration
        # Load dataset samples metadata if available, otherwise obtain it
        metadata_file = metadata_dir / f'metadata_{self.dataset_name}.json'
        try:
            with open(metadata_file,'r') as f:
                loaded_samples = json.load(f)
            self.dataset_samples = [DataSample(**sample) for sample in loaded_samples]
        except FileNotFoundError:
            self.dataset_samples = self._get_dataset_samples()
            with open(metadata_file,'w') as f:
                json.dump(
                    [sample.to_dict() for sample in self.dataset_samples],
                    f,
                    indent=4,
                )
    
    def __len__(self):
        return len(self.dataset_samples)
    
    def __getitem__(self,idx):
        """
        To maintain compatibility with previous versions,
        `ilens` and empty tensors are returned along 
        close mic mixtures and references
        """
        mix,refs = self.dataset_samples[idx].load_audio()
        ilen = torch.tensor(max(mix.shape))
        
        return mix, ilen, refs, torch.empty(0), torch.empty(0)

    def _scan_dataset_dir(self):
        dataset_perfs = dict()
        for audio_file in self.root_dir.rglob('*.wav'):
            perf_name = audio_file.parts[-2]
            file_type = audio_file.stem

            dataset_perfs.setdefault(
                perf_name,{}
            ).setdefault(
                file_type,
                audio_file
            )

        return dataset_perfs

    def _get_perf_samples(self,perf_data):
        perf_samples = []
        power_per_chunk = []
        # mix and references filepaths
        mix_filepath = perf_data.get('mixture')
        refs_filepath = perf_data.get('REFS')
        # compute the total number of signal chunks
        info = sf.info(refs_filepath)
        chunk_len = torch.ceil(
            torch.tensor(info.samplerate * self.chunk_duration)
            ).type(torch.int).item()
        stride = torch.ceil(
            torch.tensor(info.samplerate * self.stride_duration)
            ).type(torch.int).item()
        signal_len = torch.ceil(
            torch.tensor(info.samplerate * info.duration)
            ).type(torch.int).item()
        n_chunks = torch.ceil(
            torch.tensor((signal_len - chunk_len + stride)/stride)
        ).type(torch.int).item()
        # iterate over the audio chunks appending to the `perf_samples` and
        # the `power_per_chunk` lists
        for chunk_idx in range(n_chunks):
            start = chunk_idx*stride
            stop = start + chunk_len
            sample = DataSample(
                mix_filepath = mix_filepath,
                ref_filepath = refs_filepath,
                chunk_idx = chunk_idx,
                start = start,
                stop = stop,
            )
            perf_samples.append(sample)
            power_per_chunk.append(sample.power)
        # Get the power per chunk in decibels (normalized to 0 dB)
        power_per_chunk = torch.tensor(power_per_chunk)
        power_per_chunk /= power_per_chunk.max()
        power_per_chunk = 10*torch.log10(power_per_chunk)
        # Find the indices of the chunks whose power is greater than the silence threshold
        non_silent_chunk_indices = (power_per_chunk > self.silence_threshold).nonzero()
        # Return the non-silent signal chunks
        return [perf_samples[sample_idx] for sample_idx in non_silent_chunk_indices]

    def _get_dataset_samples(self):
        dataset_samples = []
        dataset_perfs = self._scan_dataset_dir()
        for perf_data in dataset_perfs.values():
            dataset_samples += self._get_perf_samples(perf_data)

        return dataset_samples