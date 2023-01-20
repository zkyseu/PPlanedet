import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from pplanedet.utils.options import parse_args
from pplanedet.utils.config import get_config
from pplanedet.utils.py_config import Config
from pplanedet.utils.setup import setup
from pplanedet.engine.trainer import Trainer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(args, cfg):
    # init environment, include logger, dynamic graph, seed, device, train or test mode...
    setup(args, cfg)
    # build trainer
    trainer = Trainer(cfg)

    # continue train or evaluate, checkpoint need contain epoch and optimizer info
    if args.resume:
        trainer.resume(args.resume)
    # evaluate or finute, only load generator weights
    elif args.load:
        trainer.load(args.load)

    if args.evaluate_only:
        trainer.val()
        return
        
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    if args.config_file.endswith('yaml') or args.config_file.endswith('yml'):
        cfg = get_config(
            args.config_file, overrides=args.override)
    elif args.config_file.endswith('py'):
        cfg = Config.fromfile(args.config_file)
    main(args, cfg)