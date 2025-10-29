import torch
import pathlib
import soundfile as sf
import json
from dataclasses import dataclass
from typing import List

@dataclass
class DataSample:
    mixture_path: str
    source_paths: List[str]
    instrument_names: List[str]

    def to_dict(self):
        return {
            'mixture_path': self.mixture_path,
            'source_paths': self.source_paths,
            'instrument_names': self.instrument_names,
        }

class AutoSourceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 subset='train',
                 chunk_duration=4.0,
                 stride_duration=2.0,
                 sample_rate=44100,
                 metadata_dir=None):
        self.root_dir = pathlib.Path(root_dir)
        self.subset = subset
        self.chunk_duration = chunk_duration
        self.stride_duration = stride_duration
        self.sample_rate = sample_rate

        self.metadata_dir = pathlib.Path(metadata_dir) if metadata_dir else self.root_dir
        self.metadata_file = self.metadata_dir / f"metadata_{subset}.json"

        # Lista de carpetas con ejemplos
        if subset is not None:
            data_dir = self.root_dir / subset
        else:
            data_dir = self.root_dir
        self.examples = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]

        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.dataset_samples = [DataSample(**sample) for sample in json.load(f)]
        else:
            self.dataset_samples = self._get_dataset_samples()
            # Puedes descomentar para guardar metadata completa
            # with open(self.metadata_file, 'w') as f:
            #     json.dump([s.to_dict() for s in self.dataset_samples], f, indent=4)

        self.chunks = []
        self._prepare_chunks()

    def _prepare_chunks(self):
        chunk_len = int(self.sample_rate * self.chunk_duration)
        stride = int(self.sample_rate * self.stride_duration if self.stride_duration else chunk_len)

        for sample in self.dataset_samples:
            mixture_path = pathlib.Path(sample.mixture_path)

            info = sf.info(mixture_path)
            track_len = info.frames
            sr = info.samplerate
            if sr != self.sample_rate:
                raise ValueError(f"Sample rate mismatch: {sr} != {self.sample_rate}")

            source_paths = [pathlib.Path(p) for p in sample.source_paths]

            for start in range(0, track_len - chunk_len + 1, stride):
                stop = start + chunk_len
                self.chunks.append({
                    'mixture_path': mixture_path,
                    'source_paths': source_paths,
                    'instrument_names': sample.instrument_names,
                    'start': start,
                    'stop': stop,
                    'sample_rate': sr
                })

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        # Cargo segmento de mixture
        mix, _ = sf.read(chunk['mixture_path'],
                         start=chunk['start'],
                         stop=chunk['stop'],
                         dtype='float32')
        mix = torch.from_numpy(mix).clone().contiguous()
        if mix.ndim == 1:
            mix = mix.unsqueeze(1)  # (samples,) -> (samples, 1)
        mix = mix.T  # (channels, samples)

        # MODIFICACIÓN PARA CANALES VACIOS: Rellenar mezcla si tiene menos de 12 canales
        if mix.shape[0] < 12:
            pad_channels = 12 - mix.shape[0]
            pad = torch.zeros((pad_channels, mix.shape[1]), dtype=mix.dtype)
            mix = torch.cat([mix, pad], dim=0)

        # Cargo segmento multicanal de fuentes (solo un archivo con múltiples canales)
        src_path = chunk['source_paths'][0]  # único archivo multicanal
        src, _ = sf.read(src_path,
                         start=chunk['start'],
                         stop=chunk['stop'],
                         dtype='float32')
        src = torch.from_numpy(src).clone().contiguous()

        if src.ndim == 1:
            src = src.unsqueeze(0)  # si mono, agregar dimensión canales
        else:
            src = src.T  # transponer a [channels, samples]

        # MODIFICACIÓN PARA CANALES VACIOS: Rellenar fuentes si tienen menos de 12 canales
        if src.shape[0] < 12:
            pad_src = torch.zeros((12 - src.shape[0], src.shape[1]), dtype=src.dtype)
            src = torch.cat([src, pad_src], dim=0)

        sources = src

        # MODIFICACIÓN PARA CANALES VACIOS: Rellenar nombres de instrumentos si faltan
        instrument_names = chunk['instrument_names']
        if len(instrument_names) < 12:
            instrument_names += [''] * (12 - len(instrument_names))

        # Comprobamos que las formas coincidan (C, T)
        assert mix.shape == sources.shape, f"Shape mismatch: mix {mix.shape} vs sources {sources.shape}"

        # Devuelvo también la lista de nombres de instrumentos
        return mix.contiguous(), sources.contiguous(), instrument_names

    def _get_dataset_samples(self):
        samples = []
        for folder in self.examples:  # combinacionX
            # Buscar carpetas aleatorias
            random_dirs = [d for d in folder.iterdir() if d.is_dir()]
            if not random_dirs:
                continue

            for random_dir in random_dirs:
                # Buscar todas las subcarpetas dentro de la carpeta aleatoria
                subfolders = [sf for sf in random_dir.iterdir() if sf.is_dir()]
                if not subfolders:
                    continue

                for subfolder in subfolders:
                    # Buscar mixture y refs sin importar mayúsculas
                    mixture_path = None
                    refs_path = None
                    for f in subfolder.glob("*"):
                        if f.is_file():
                            name_lower = f.name.lower()
                            if "mixture" in name_lower and name_lower.endswith(".wav"):
                                mixture_path = f
                            elif "refs" in name_lower and name_lower.endswith(".wav"):
                                refs_path = f

                    if not mixture_path or not refs_path:
                        print(f"No se encontró mixture.wav o refs.wav en {subfolder}")
                        continue

                    # Buscar JSON en la carpeta aleatoria
                    random_json_files = [f for f in folder.glob("*.json") if f.name != "metadata.json"]
                    if not random_json_files:
                        print(f"No random json file found in {folder}")
                        continue

                    with open(random_json_files[0], 'r') as f:
                        random_json_data = json.load(f)

                    if "channel_mapper" not in random_json_data:
                        print(f"No channel_mapper found in {random_json_files[0]}")
                        continue

                    instrument_names = list(random_json_data["channel_mapper"].keys())

                    sample = DataSample(
                        mixture_path=str(mixture_path),
                        source_paths=[str(refs_path)],
                        instrument_names=instrument_names
                    )
                    samples.append(sample)

                    # Guardar metadata individual
                    metadata_path = random_dir / "metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(sample.to_dict(), f, indent=4)

        return samples


