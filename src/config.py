import sched
from dataclasses import dataclass, field

from sklearn import base


@dataclass
class Common:
    _name: str = 'default'
    seed: int = 39
    device: str = 'cuda'


@dataclass
class FrontendModel:
    _name: str = 'wav2vec2-conformer-base'
    path: str = 'path/to/wav2vec2-conformer-base'
    freeze: bool = True


@dataclass
class Model:
    _name: str = 'emotion-wav2vec2-conformer'
    frontend_model: FrontendModel = FrontendModel()


@dataclass
class Dataset:
    _name: str = 'IEMOCAP'
    d_path: str = 'path/to/dataset/IEMOCAP_full_release'
    ratio: list = field(default_factory=lambda: [0.8, 0.1, 0.1])
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class Train:
    fold: int = 5
    epoch: int = 100


@dataclass
class Optimizer:
    _name: str = 'adam'
    lr: float = 5e-4
    weight_decay: float = 1e-4


@dataclass
class Scheduler:
    _name: str = 'cosine'
    T_max: int = 100
    eta_min: float = 1e-5
    last_epoch: int = -1


@dataclass
class Config:
    common: Common = Common()
    model: Model = Model()
    dataset: Dataset = Dataset()
    train: Train = Train()
    optimizer: Optimizer = Optimizer()
    scheduler: Scheduler = Scheduler()