import os, argparse
import torch
import pytorch_lightning as pl
from train_utils import args_from_yaml
from data_utils import atmDataModule

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-C', type=str, default='fixmatch_1.yaml')
    parser.add_argument('--tqdm_rate', '-TQDM', type=int, default=1)
    args = parser.parse_args()

    # overwrite args
    args.config = os.path.join('config', args.config)
    args_from_yaml(args=args, yml=args.config)

    # device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # seed
    pl.seed_everything(args.seed)

    # build dataset
    dm = atmDataModule(args=args)