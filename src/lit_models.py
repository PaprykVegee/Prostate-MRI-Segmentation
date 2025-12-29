import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torchmetrics.classification import MulticlassJaccardIndex

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from models import VNet, AttentionVNet

class LitBaseVNet(pl.LightningModule):
    def __init__(self, model_obj, in_ch: int, out_ch: int, lr: float = 1e-3):
        super().__init__()

        # model_obj to już zainicjalizowana instancja (np. VNet(1,3))
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

    def compute_loss_and_metrics(self, batch):
        """
        Metoda pomocnicza do testów i kroków walidacyjnych.
        """
        x = batch["image"]
        y = batch["label"].long()

        logits = self(x)
        loss = self.loss_fn(logits, y)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)

            y_oh = [AsDiscrete(to_onehot=self.hparams.out_ch)(yy) for yy in decollate_batch(y)]
            p_oh = [AsDiscrete(argmax=True, to_onehot=self.hparams.out_ch)(pp) for pp in decollate_batch(logits)]

            self.dice_metric(y_pred=p_oh, y=y_oh)
            dice = self.dice_metric.aggregate().item()
            self.dice_metric.reset() 

            miou = self.miou_metric(preds, y.squeeze(1))

        return loss, miou, dice

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"].long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, miou, dice = self.compute_loss_and_metrics(batch)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_epoch=True, prog_bar=True)
        self.log("val_miou", miou, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, miou, dice = self.compute_loss_and_metrics(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_dice", dice, on_epoch=True, prog_bar=True)
        self.log("test_miou", miou, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# ==========================================================
# TEST POPRAWNOŚCI
# ==========================================================
if __name__ == "__main__":
    # 1. Sprawdzenie zwykłego VNeta
    print("--- Test: VNet ---")
    vnet = VNet(in_ch=1, out_ch=3)
    lit_model = LitBaseVNet(model_obj=vnet, in_ch=1, out_ch=3, lr=1e-3)

    dummy_image = torch.randn(2, 1, 32, 32, 32) 
    dummy_label = torch.randint(0, 3, (2, 1, 32, 32, 32)) 
    batch = {"image": dummy_image, "label": dummy_label}

    loss, miou, dice = lit_model.compute_loss_and_metrics(batch)
    print(f"Loss: {loss.item():.4f}, mIoU: {miou:.4f}, Dice: {dice:.4f}")

    # 2. Sprawdzenie VNeta z Atencją
    print("\n--- Test: AttentionVNet ---")
    att_vnet = AttentionVNet(in_ch=1, out_ch=3)
    lit_att_model = LitBaseVNet(model_obj=att_vnet, in_ch=1, out_ch=3, lr=1e-3)

    loss_att, miou_att, dice_att = lit_att_model.compute_loss_and_metrics(batch)
    print(f"Loss: {loss_att.item():.4f}, mIoU: {miou_att:.4f}, Dice: {dice_att:.4f}")