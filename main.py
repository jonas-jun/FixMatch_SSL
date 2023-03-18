import os, argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from train_utils import args_from_yaml
from data_utils import atmDataModule
from model import atmNet

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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
    print('Data Module Built')

    # build model
    if args.mode == 'train':
        model = None
    elif args.mode == 'test':
        model = atmNet.load

    # lightning callbacks
    ckpt_callback = ModelCheckpoint(
        monitor='val_acc_top1',
        dirpath=os.path.join(args.output_dir, args.experiment),
        filename='{epoch:02d}-{val_acc_top1:.2f}-{val_acc_top5:.2f}-{val_f1_macro:.2f}',
        save_top_k=3,
        save_weights_only=True,
        mode='max',
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    pbar = TQDMProgressBar(refresh_rate=args.tqdm_rate)
    tb_logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.experiment)

    # trainer
    precision = 16 if args.amp else 32
    if args.mode == 'train':
        trainer = pl.Trainer(
            accelerator='gpu',
            strategy='ddp',
            devices=2,
            logger=[tb_logger],
            default_root_dir=args.output_dir,
            callbacks=[ckpt_callback, lr_monitor, pbar],
            max_epochs=args.n_epochs,
            log_every_n_steps=100,
            precision=precision
        )
    elif args.mode == 'test':
        trainer = pl.Trainer(
            accelerator='gpu',
            precision=precision,
            max_epochs=1
        )

    # train
    if args.mode == 'train':
        trainer.fit(model=model, datamodule=dm)
    elif args.mode == 'test':
        trainer.test(model=model, datamodule=dm)