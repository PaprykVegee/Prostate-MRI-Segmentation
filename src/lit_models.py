import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

class LitBaseVNet(pl.LightningModule):
    def __init__(self, model_obj, in_ch: int, out_ch: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model_obj'])
        self.model = model_obj 

        self.loss_fn = DiceCELoss(
            to_onehot_y=True, 
            softmax=True, 
            include_background=True,
            lambda_dice=1.0,
            lambda_ce=1.0
        )

        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.miou_metric = MulticlassJaccardIndex(num_classes=out_ch, average="macro")

    def forward(self, x):
        return self.model(x)

    def _prepare_batch(self, batch):
        """
        Prywatna metoda do bezpiecznego wypakowania danych z batcha.
        Rozwiązuje błąd: 'list indices must be integers or slices, not str'
        """
        if isinstance(batch, list):
            batch = batch[0]
        
        x = batch["image"]
        y = batch["label"].long()
        
        # Konwersja MetaTensor -> Tensor (rozwiązuje błąd 0-d array)
        if hasattr(x, "as_tensor"): x = x.as_tensor()
        if hasattr(y, "as_tensor"): y = y.as_tensor()
            
        return x, y

    def compute_loss_and_metrics(self, batch):
        x, y = self._prepare_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)

        with torch.no_grad():
            # Inicjalizacja transformacji post-processingowych
            post_label = AsDiscrete(to_onehot=self.hparams.out_ch)
            post_pred = AsDiscrete(argmax=True, to_onehot=self.hparams.out_ch)

            # Rozpakowanie batcha do obliczeń metryk (na czystych tensorach)
            y_list = decollate_batch(y)
            p_list = decollate_batch(logits)

            y_oh = [post_label(yy) for yy in y_list]
            p_oh = [post_pred(pp) for pp in p_list]

            # Obliczanie Dice
            self.dice_metric(y_pred=p_oh, y=y_oh)
            dice = self.dice_metric.aggregate().item()
            self.dice_metric.reset()

            # Obliczanie mIoU (MulticlassJaccardIndex)
            preds = torch.argmax(logits, dim=1)
            # Y może mieć kształt [B, 1, H, W, D], mIoU chce [B, H, W, D]
            miou = self.miou_metric(preds, y.squeeze(1))

        return loss, miou, dice

    def training_step(self, batch, batch_idx):
        x, y = self._prepare_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, miou, dice = self.compute_loss_and_metrics(batch)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_epoch=True, prog_bar=True)
        self.log("val_miou", miou, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, miou, dice = self.compute_loss_and_metrics(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_dice", dice, on_epoch=True, prog_bar=True)
        self.log("test_miou", miou, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # T_max=50, bo tyle ustawiasz w Trainerze (max_epochs=50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}