import time
import gzip
import torch
import threading
import numpy as np
import pandas as pd


from tqdm.auto import tqdm
from pathlib import Path
from gzip import GzipFile
from multiprocessing import Pool
from omegaconf import DictConfig
from imops import crop_to_box, zoom
from functools import cached_property, partial
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple, Sequence, Dict, NamedTuple


from cotomka.datasets.base import Dataset
from cotomka.preprocessing.common import get_body_box, mask_to_bbox, rescale_hu_piecewise
from cotomka.utils.io import load_numpy, load_json, save_numpy, save_json
from cotomka.utils.data_prefetcher import DataPrefetcher

from nnssl.data.dataloading.dataset import nnSSLDatasetBlosc2


class PreprocessingConfig(NamedTuple):
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    hu_pivots: Sequence[float] = (-1000.0, -200.0, 200.0, 1500.0)
    rescaled_pivots: Sequence[float] = (0.0, 0.33, 0.67, 1.0)
    min_image_size: Tuple[int, int, int] = (160, 160, 160)

    @classmethod
    def from_dict_config(cls, config: DictConfig) -> 'PreprocessingConfig':
        return cls(
            voxel_spacing=tuple(config.voxel_spacing),
            hu_pivots=tuple(config.hu_pivots),
            rescaled_pivots=tuple(config.rescaled_pivots),
            min_image_size=tuple(config.min_image_size),
        )


class AbdomenAtlasPreprocessed(Dataset):
    name = 'dinov2_3d/abdomen_atlas'

    def __init__(self, index=None):
        super().__init__()
        self.init_index = index

    def _get_image(self, id: str) -> np.ndarray:
        return load_numpy(self.root_dir / id / 'image.npy.gz', decompress=True).astype('float32')
    
    def _get_labels(self, id: str):
        return np.random.randint(0, 10)

    def _get_voxel_spacing(self, id: str) -> Tuple[float, float, float]:
        return tuple(load_json(self.root_dir / id / 'voxel_spacing.json'))

    def _get_mask(self, id: str) -> np.ndarray:
        return load_numpy(self.root_dir / id / 'mask.npy.gz', decompress=True)

    def prepare(self, raw: Dataset, preprocessing_config: PreprocessingConfig, num_workers: int = 8) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        save_json(preprocessing_config.voxel_spacing, self.root_dir / 'voxel_spacing.json')

        _prepare_image = partial(self._prepare_image, raw=raw, preprocessing_config=preprocessing_config)
        with Pool(num_workers) as p:
            _ = list(tqdm(p.imap(_prepare_image, raw.ids), total=len(raw.ids)))

    def _prepare_image(self, id: str, raw: Dataset, preprocessing_config: PreprocessingConfig) -> Dict[str, np.ndarray]:
        data = raw.get(id, fields=['image', 'voxel_spacing'])

        # crop to body
        box = get_body_box(data['image'], data['voxel_spacing'])
        image = crop_to_box(data['image'], box, num_threads=-1, backend='Scipy')

        # zoom to config.voxel_spacing
        image = image.astype('float32')
        scale_factor = tuple(data['voxel_spacing'][i] / preprocessing_config.voxel_spacing[i] for i in range(3))
        image = zoom(image, scale_factor, fill_value=np.min, num_threads=-1, backend='Scipy')

        # zoom may pad image with zeros
        box = mask_to_bbox(image > image.min())
        image = crop_to_box(image, box, num_threads=-1, backend='Scipy')

        if any(image.shape[i] < preprocessing_config.min_image_size[i] for i in range(3)):
            return

        # rescale Hounsfield Units (HU) using piecewise-linear func
        image = rescale_hu_piecewise(image, preprocessing_config.hu_pivots, preprocessing_config.rescaled_pivots)

        data_dir = self.root_dir / _cut_ext(id)
        data_dir.mkdir()
        save_numpy(image.astype('float16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)


class AMOSCTUnlabeledTrainPreprocessed(Dataset):
    name = 'dinov2_3d/amos_train'

    def __init__(self, index=None):
        super().__init__()
        self.init_index = index

    def _get_image(self, id: str) -> np.ndarray:
        return load_numpy(self.root_dir / id / 'image.npy.gz', decompress=True).astype('float32')
    
    def _get_labels(self, id: str):
        return np.random.randint(0, 10)
        
    def _get_voxel_spacing(self, id: str) -> Tuple[float, float, float]:
        return tuple(load_json(self.root_dir / id / 'voxel_spacing.json'))

    def prepare(self, raw: Dataset, preprocessing_config: PreprocessingConfig, num_workers: int = 8) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        save_json(preprocessing_config.voxel_spacing, self.root_dir / 'voxel_spacing.json')

        _prepare_image = partial(self._prepare_image, raw=raw, preprocessing_config=preprocessing_config)
        with Pool(num_workers) as p:
            _ = list(tqdm(p.imap(_prepare_image, raw.ids), total=len(raw.ids)))

    def _prepare_image(self, id: str, raw: Dataset, preprocessing_config: PreprocessingConfig) -> Dict[str, np.ndarray]:
        data = raw.get(id, fields=['image', 'voxel_spacing'])

        # crop to body
        box = get_body_box(data['image'], data['voxel_spacing'])
        image = crop_to_box(data['image'], box, num_threads=-1, backend='Scipy')

        # zoom to config.voxel_spacing
        image = image.astype('float32')
        scale_factor = tuple(data['voxel_spacing'][i] / preprocessing_config.voxel_spacing[i] for i in range(3))
        image = zoom(image, scale_factor, fill_value=np.min, num_threads=-1, backend='Scipy')

        # zoom may pad image with zeros
        box = mask_to_bbox(image > image.min())
        image = crop_to_box(image, box, num_threads=-1, backend='Scipy')

        if any(image.shape[i] < preprocessing_config.min_image_size[i] for i in range(3)):
            return

        # rescale Hounsfield Units (HU) using piecewise-linear func
        image = rescale_hu_piecewise(image, preprocessing_config.hu_pivots, preprocessing_config.rescaled_pivots)

        data_dir = self.root_dir / _cut_ext(id)
        data_dir.mkdir()
        save_numpy(image.astype('float16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)


def _cut_ext(filename: str) -> str:
    return filename.split('.')[0]

class CTRATEPreprocessed(Dataset):
    name = 'dinov2_3d/ct_rate'

    def __init__(self, ids=None):
        super().__init__()
        self.init_ids = ids

    @cached_property
    def _labels_df(self):
        return pd.read_csv(self.root_dir / 'labels.csv', index_col='VolumeName')

    @cached_property
    def _metadata_df(self):
        return pd.read_csv(self.root_dir / 'metadata.csv', index_col='VolumeName')

    @cached_property
    def _reports_df(self):
        return pd.read_csv(self.root_dir / 'reports.csv', index_col='VolumeName')

    @cached_property
    def ids(self) -> Tuple[str]:
        return tuple(sorted(id for id in self._labels_df.index if (self.root_dir / _cut_ext(id)).exists()))

    def _get_image(self, id: str) -> np.ndarray:
        return load_numpy(self.root_dir / _cut_ext(id) / 'image.npy.gz', decompress=True)

    def _get_voxel_spacing(self, id: str) -> Tuple[float, float, float]:
        return load_json(self.root_dir / 'voxel_spacing.json')

    def _get_findings(self, id: str) -> str:
        return self._reports_df.loc[id, 'Findings_EN']

    def _get_impression(self, id: str) -> str:
        return self._reports_df.loc[id, 'Impressions_EN']

    def _get_labels(self, id: str) -> Dict[str, int]:
        return self._labels_df.loc[id].to_dict()

    def _get_patient_id(self, id: str) -> str:
        return '_'.join(id.split('_')[:2])

    def prepare(self, raw: Dataset, preprocessing_config: PreprocessingConfig, num_workers: int = 8) -> None:
        if self.root_dir.exists():
            raise OSError(f'Directory {self.root_dir} already exists')
        self.root_dir.mkdir(parents=True)

        raw.labels_df.to_csv(self.root_dir / 'labels.csv', index=True)
        raw.metadata_df.to_csv(self.root_dir / 'metadata.csv', index=True)
        raw.reports_df.to_csv(self.root_dir / 'reports.csv', index=True)

        save_json(preprocessing_config.voxel_spacing, self.root_dir / 'voxel_spacing.json')

        prepare_image = partial(self._prepare_image, raw=raw, preprocessing_config=preprocessing_config)
        with Pool(num_workers) as p:
            _ = list(tqdm(p.imap(prepare_image, raw.ids), total=len(raw.ids)))

    def _prepare_image(self, id: str, raw: Dataset, preprocessing_config: PreprocessingConfig) -> Dict[str, np.ndarray]:
        data = raw.get(id, fields=['image', 'voxel_spacing'])

        # crop to body
        box = get_body_box(data['image'], data['voxel_spacing'])
        image = crop_to_box(data['image'], box, num_threads=-1, backend='Scipy')

        # zoom to config.voxel_spacing
        image = image.astype('float32')
        scale_factor = tuple(data['voxel_spacing'][i] / preprocessing_config.voxel_spacing[i] for i in range(3))
        image = zoom(image, scale_factor, fill_value=np.min, num_threads=-1, backend='Scipy')

        # zoom may pad image with zeros
        box = mask_to_bbox(image > image.min())
        image = crop_to_box(image, box, num_threads=-1, backend='Scipy')

        if any(image.shape[i] < preprocessing_config.min_image_size[i] for i in range(3)):
            return

        # rescale Hounsfield Units (HU) using piecewise-linear func
        image = rescale_hu_piecewise(image, preprocessing_config.hu_pivots, preprocessing_config.rescaled_pivots)

        data_dir = self.root_dir / _cut_ext(id)
        data_dir.mkdir()
        save_numpy(image.astype('float16'), data_dir / 'image.npy.gz', compression=1, timestamp=0)


class CTRATETrainPreprocessed(CTRATEPreprocessed):
    name = 'dinov2_3d/ct_rate_train'

class CTRATEValPreprocessed(CTRATEPreprocessed):
    name = 'dinov2_3d/ct_rate_val'


class OpenNeuroPreprocessed(nnSSLDatasetBlosc2):

    @cached_property
    def ids(self) -> Tuple[str]:
        _exclude_indices = [58892]
        _exclude_ids = [self.image_identifiers[idx] for idx in _exclude_indices]
        filtered_ids = [x for x in self.image_identifiers if x not in _exclude_ids]
        return filtered_ids
    
    def get(self, id, fields):
        sub_path = self.image_dataset[id].get_output_path("image", ext=".b2nd")
        out_fpath = Path("/home/jovyan/misha/misc/cotomka/Dataset745_OpenMind/nnsslPlans_onemmiso/") / (sub_path[:-5] + ".npy.gz")
        # out_fpath = out_fpath.parent / f"{out_fpath.name[:-7]}.npy.gz"
        with gzip.open(out_fpath, "rb") as f:
            image = np.load(f, allow_pickle=True)
        return {"image" : image}


class BraTSClassificationTrain:
    def __init__(self):
        self.root_path = Path("/home/jovyan/datasets/BraTS-MEN-Train/train")
    
    @cached_property
    def ids(self) -> Tuple[str]:
        return list([Path(x.parent.name) / x.name for x in self.root_path.rglob("*.npy.gz")])[50:-50]
    
    def get(self, id, fields):
        path = self.root_path / id
        ret = {}
        if "image" in fields:
            with gzip.open(path, "rb") as f:
                image = np.load(f, allow_pickle=True)
            d, h, w = image.shape
            dl = max((d - 128) // 2, 0)
            hl = max((h - 128) // 2, 0)
            wl = max((w - 128) // 2, 0)
            image = image[dl:dl+128, hl:hl+128, wl:wl+128]
            mean = image.mean()
            std = image.std()
            ret["image"] = (image - mean) / (std + 1e-8)
        if "labels" in fields:
            labels = {"grade1" : 1, "grade2" : 0} if "Grade1" in str(id) else {"grade1" : 0, "grade2" : 1}
            ret["labels"] = labels
        return ret
    

class BraTSClassificationVal:
    def __init__(self):
        self.root_path = Path("/home/jovyan/datasets/BraTS-MEN-Train/train")
    
    @cached_property
    def ids(self) -> Tuple[str]:
        return list([Path(x.parent.name) / x.name for x in self.root_path.rglob("*.npy.gz")])[:50] + \
            list([Path(x.parent.name) / x.name for x in self.root_path.rglob("*.npy.gz")])[-50:]
    
    def get(self, id, fields):
        path = self.root_path / id
        ret = {}
        if "image" in fields:
            with gzip.open(path, "rb") as f:
                image = np.load(f, allow_pickle=True)
            d, h, w = image.shape
            dl = max((d - 128) // 2, 0)
            hl = max((h - 128) // 2, 0)
            wl = max((w - 128) // 2, 0)
            image = image[dl:dl+128, hl:hl+128, wl:wl+128]
            mean = image.mean()
            std = image.std()
            ret["image"] = (image - mean) / (std + 1e-8)
        if "labels" in fields:
            labels = {"grade1" : 1, "grade2" : 0} if "Grade1" in str(id) else {"grade1" : 0, "grade2" : 1}
            ret["labels"] = labels
        return ret


class MedicalImageDatasetDINO(TorchDataset):
    def __init__(
        self,
        sources,
        transform=None,
        prefetch=False,
        replication=None,
        prefetch_workers=4,
        prefetch_buffer_size=256,
        backend="threading",
        fields=['image']
    ):
        self.datasets = sources
        self.transform = transform
        self.prefetch = prefetch
        self.backend = backend
        self.prefetch_buffer_size = prefetch_buffer_size
        self.prefetch_workers = prefetch_workers
        self.shift_locks = [threading.Lock() for _ in sources]
        self.shifts = [0] * len(sources)
        self.fields = fields
        
        self.sum_lens = np.cumsum([len(x.ids)-1 for x in sources])
        source_weights = np.array([len(x.ids)-1 for x in sources])
        self.source_weights = source_weights / sum(source_weights)
        self.index = []
        for source in sources:
            self.index.extend(source.ids[:-1])

        if prefetch:
            self.prefetchers = [
                DataPrefetcher(
                    source,
                    int(prefetch_workers * self.source_weights[i])+1,
                    int(prefetch_buffer_size * self.source_weights[i]),
                    replication,
                    backend=backend,
                    fields=fields
                ) for i, source in enumerate(sources)
            ]
        else:
            self.prefetchers = []

    def __len__(self):
        return sum([len(x.ids)-1 for x in self.datasets])
    
    def get_source(self, idx):
        if idx < 0:
            idx = self.__len__() + idx
        return self.datasets[(idx >= self.sum_lens).sum()]

    def __getitem__(self, idx=None):
        if self.prefetch:
            prefetcher = np.random.choice(self.prefetchers, p=self.source_weights)
            x = next(prefetcher)
            if self.transform:
                x = self.transform(x)
        else:
            source = self.get_source(idx)
            index = self.index[idx]
            x = source.get(index, self.fields)
            if self.transform:
                x = self.transform(x)

        return x
    
    def destroy(self):
        for prefetcher in self.prefetchers:
            prefetcher.destroy()
    
    def __del__(self):
        self.destroy()