import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from train_utils import get_metrics, ce_loss, consistency_loss

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
        self.metric_single, _ = get_metrics(n_classes=self.args.n_classes)

    def set_model_timm(self):
        import models.from_timm as timm
        if 'conv' in self.experiment:
            return timm.load_convnext()
        
    def forward(self, images):
        return self.model(images)
    
    def cost_fn(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s, T, p_cutoff):
        sup_loss = ce_loss(logits=logits_x_lb, targets=y_lb, reduction='mean')
        unsup_loss, mask, select, pseudo_label, max_probs_mean = consistency_loss(
                                    logits_w=logits_x_ulb_w,
                                    logits_s=logits_x_ulb_s,
                                    name='ce',
                                    T=T,
                                    p_cutoff=p_cutoff,
                                    use_hard_label=self.args.hard_label)
        total_loss = sup_loss + self.args.lambda_u * unsup_loss
        num_no_masked = len(torch.nonzero(select))
        self.log('not_masked_ratio', num_no_masked / len(select), prog_bar=True)
        return sup_loss, unsup_loss, total_loss

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
        batch_label, batch_unlabel = batch['label'], batch['unlabel']
        (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s) = batch_label, batch_unlabel
        num_lb, num_ulb = x_lb.shape[0], x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]
        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        logits = self(inputs)
        logits_x_lb = logits[:num_lb]
        logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        sup_loss, unsup_loss, total_loss = self.cost_fn(logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s, T=self.args.T, p_cutoff=self.args.p_cutoff)
        
        self.log('Loss_Supervised', sup_loss, prog_bar=True)
        self.log('Loss_UnSupervised', unsup_loss, prog_bar=True)
        self.log('Loss_Total', total_loss, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        self.shared_eval_step(batch=batch)

    def test_step(self, batch, batch_idx):
        self.shared_eval_step(batch=batch)

    def on_validation_epoch_end(self):
        self.shared_eval_epoch_end(mode='val')
    
    def on_test_epoch_end(self) -> None:
        self.shared_eval_epoch_end(mode='test')

    def shared_eval_step(self, batch):
        idx, x_val, y_val = batch
        logits = self(x_val)
        self.metric_single.update(logits, y_val)
    
    def shared_eval_epoch_end(self, mode):
        metric_out = self.metric_single.compute()
        self.torchmetric_logger(result_dict=metric_out, mode=mode)
        self.metric_single.reset()

    def torchmetric_logger(self, result_dict, mode, multilabel=False):
        if not multilabel:
            self.log('{}_acc_top1'.format(mode), result_dict['acc_top1'].cpu().item(), sync_dist=True)
            self.log('{}_acc_top5'.format(mode), result_dict['acc_top5'].cpu().item(), sync_dist=True)
            self.log('{}_precision'.format(mode), result_dict['precision'].cpu().item(), sync_dist=True)
            self.log('{}_recall'.format(mode), result_dict['recall'].cpu().item(), sync_dist=True)
            self.log('{}_f1_macro'.format(mode), result_dict['f1'].cpu().item(), sync_dist=True)
        else:
            self.log('{}_Multi Label Precision4'.format(mode), result_dict['multi_precision_4'].cpu().item(), sync_dist=True)
            self.log('{}_Multi Label Precision5'.format(mode), result_dict['multi_precision_5'].cpu().item(), sync_dist=True)
            self.log('{}_Multi Label Precision6'.format(mode), result_dict['multi_precision_6'].cpu().item(), sync_dist=True)
            self.log('{}_Multi Label Recall4'.format(mode), result_dict['multi_recall_4'].cpu().item(), sync_dist=True)
            self.log('{}_Multi Label Recall5'.format(mode), result_dict['multi_recall_5'].cpu().item(), sync_dist=True)
            self.log('{}_Multi Label Recall6'.format(mode), result_dict['multi_recall_6'].cpu().item(), sync_dist=True)