import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.metrics import compute_dice
import numpy as np

class LitBaseVNet(pl.LightningModule):
    def __init__(self, model_obj, in_ch: int, out_ch: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['model_obj'])
        self.model = model_obj 

        #dodane
        self.register_buffer("ce_weights", torch.tensor([0.2, 1.0, 1.5]))
        # 

        self.loss_fn = DiceCELoss(
            to_onehot_y=True, 
            softmax=True, 
            # include_background=True,
            include_background=False,
            lambda_dice=2.0,
            lambda_ce=0.5,
            # jak będzie loss skakał to zmienić np dice na 0.5 czy coś
            weight=self.ce_weights #dodane
        )

        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.miou_metric = MulticlassJaccardIndex(num_classes=out_ch, average="macro")

        self.post_label = AsDiscrete(to_onehot=out_ch)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=out_ch)

        
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

    # def compute_loss_and_metrics(self, batch):
    #     x, y = self._prepare_batch(batch)
    #     logits = self(x)
    #     loss = self.loss_fn(logits, y)

    #     with torch.no_grad():
    #         # Inicjalizacja transformacji post-processingowych
    #         # post_label = AsDiscrete(to_onehot=self.hparams.out_ch)
    #         # post_pred = AsDiscrete(argmax=True, to_onehot=self.hparams.out_ch)

    #         # Rozpakowanie batcha do obliczeń metryk (na czystych tensorach)
    #         y_list = decollate_batch(y)
    #         p_list = decollate_batch(logits)

    #         y_oh = [self.post_label(yy) for yy in y_list]
    #         p_oh = [self.post_pred(pp) for pp in p_list]

    #         # Obliczanie Dice
    #         self.dice_metric(y_pred=p_oh, y=y_oh)
    #         dice = self.dice_metric.aggregate().item()
    #         self.dice_metric.reset()

    #         # Obliczanie mIoU (MulticlassJaccardIndex)
    #         preds = torch.argmax(logits, dim=1)
    #         # Y może mieć kształt [B, 1, H, W, D], mIoU chce [B, H, W, D]
    #         miou = self.miou_metric(preds, y.squeeze(1))

    #     return loss, miou, dice



    def compute_loss_and_metrics(self, batch):
        x, y = self._prepare_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)

        with torch.no_grad():
            y_list = decollate_batch(y)
            p_list = decollate_batch(logits)

            y_oh = torch.stack([self.post_label(yy) for yy in y_list])
            p_oh = torch.stack([self.post_pred(pp) for pp in p_list])

            # Obliczanie Dice
            dice_per_class = compute_dice(y_pred=p_oh, y=y_oh, include_background=False)
            
            # Zamiast nanmean, wypełniamy NaN zerami
            # (NaN pojawia się, gdy klasa nie występuje w labelu LUB predykcji)
            dice_per_class = torch.where(torch.isnan(dice_per_class), torch.zeros_like(dice_per_class), dice_per_class)
            
            # Średnia po batchu dla każdej klasy
            dice_vals = torch.mean(dice_per_class, dim=0) 
            
            # Ogólny dice (średnia z klas) - upewniamy się, że nie jest NaN
            dice_mean = torch.mean(dice_vals).item()
            if np.isnan(dice_mean):
                dice_mean = 0.0

        return loss, dice_mean, dice_vals

    def training_step(self, batch, batch_idx):
        x, y = self._prepare_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     loss, miou, dice = self.compute_loss_and_metrics(batch)
        
    #     self.log("val_loss", loss, on_epoch=True, prog_bar=True)
    #     self.log("val_dice", dice, on_epoch=True, prog_bar=True)
    #     self.log("val_miou", miou, on_epoch=True, prog_bar=True)
    #     return loss

    def validation_step(self, batch, batch_idx):
        loss, dice_mean, dice_vals = self.compute_loss_and_metrics(batch)
        
        # Logowanie główne
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice_mean, on_epoch=True, prog_bar=True)
        
        # Logowanie poszczególnych klas (zakładając out_ch=3 i include_background=False)
        # Klasa 1: Peripheral Zone (PZ), Klasa 2: Transition Zone (TZ)
        if len(dice_vals) >= 2:
            self.log("val_dice_PZ", dice_vals[0], on_epoch=True, prog_bar=False)
            self.log("val_dice_TZ", dice_vals[1], on_epoch=True, prog_bar=False)
            
        return loss

    def test_step(self, batch, batch_idx):
        loss, miou, dice = self.compute_loss_and_metrics(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_dice", dice, on_epoch=True, prog_bar=True)
        self.log("test_miou", miou, on_epoch=True, prog_bar=True)
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    #     # T_max=50, bo tyle ustawiasz w Trainerze (max_epochs=50)
    #     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-7)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=25)
    #     # return {"optimizer": optimizer, "lr_scheduler": scheduler}
    #     return {
    #     "optimizer": optimizer,
    #     "lr_scheduler": {
    #         "scheduler": scheduler,
    #         "monitor": "val_dice",  # Trzeba obserwować val_dice
    #         "interval": "epoch",
    #         "frequency": 1
    #     },
    # }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        # T_0 - po ilu epokach pierwszy restart (np. 50)
        # T_mult - czy każdy kolejny cykl ma być dłuższy (2 oznacza: 50, 100, 200...)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=50,      # Długość pierwszego cyklu
            T_mult=2,    # Każdy kolejny cykl jest 2x dłuższy
            eta_min=1e-6 # Minimalny LR, do którego schodzimy w cyklu
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # Bardzo ważne dla tego schedulera
                "frequency": 1
            },
    }