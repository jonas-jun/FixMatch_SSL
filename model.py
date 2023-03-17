import pytorch_lightning as pl

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
        return
    
    def training_step(self, batch, batch_idx):
        return
    