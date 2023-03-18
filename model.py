import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from train_utils import get_metrics

class atmNet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.experiment = args.experiment
        self.model = None

        if 'pf' in self.experiment:
            pass
        elif 'timm' in self.experiment:
            self.model = self.set_model_timm()

    def set_model_timm(self):
        import models.from_timm as timm
        if 'conv' in self.experiment:
            return timm.load_convnext()
        
    def forward(self, images):
        return self.model(images)
    
    def cost_fn(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s, T, p_cutoff):
        pass

    def configure_optimizers(self):
        if self.args.optim.upper() == 'SGD':
            optimizer = optim.SGD(self.parameters(),
                                  lr=self.args.lr,
                                  momentum=self.args.momentum,
                                  weight_decay=self.args.wdecay,
                                  nesterov=True)
        if self.args.scheduler == 'multistep':
            schduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=range(10), gamma=0.5)
        if schduler:
            return [optimizer], [schduler]
        else:
            return optimizer
    
    def training_step(self, batch, batch_idx):
        return
    