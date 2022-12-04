from paddleseg.cvlibs import Config
import paddle
from typing import Dict,Any

class Config_(Config):

    @property
    def train_unlabel_dataset_config(self) -> Dict:
        return self.dic.get('unlabel_dataset', {}).copy()

    @property
    def train_unlabel_dataset(self) -> paddle.io.Dataset:
        _train_dataset = self.train_unlabel_dataset_config
        # print(_train_dataset)
        if not _train_dataset:
            return None
        return self._load_object(_train_dataset)

    def _load_object(self, cfg: dict) -> Any:
        cfg = cfg.copy()
        if 'type' not in cfg:
            raise RuntimeError('No object information in {}.'.format(cfg))

        component = self._load_component(cfg.pop('type'))

        params = {}
        for key, val in cfg.items():
            if self._is_meta_type(val):
                params[key] = self._load_object(val)
            elif isinstance(val, list):
                params[key] = [
                    self._load_object(item)
                    if self._is_meta_type(item) else item for item in val
                ]
            else:
                params[key] = val
        return component(**params)
