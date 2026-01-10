import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import VNet, AttentionVNet, AttentionVNet 
from lit_models import LitBaseVNet

import torch


print("--- Test: VNet ---")
vnet = VNet(in_ch=1, out_ch=3)
lit_model = LitBaseVNet(model_obj=vnet, in_ch=1, out_ch=3, lr=1e-3)

dummy_image = torch.randn(2, 1, 32, 32, 32) 
dummy_label = torch.randint(0, 3, (2, 1, 32, 32, 32)) 
batch = {"image": dummy_image, "label": dummy_label}

loss, miou, dice = lit_model.compute_loss_and_metrics(batch)
print(f"Loss: {loss.item():.4f}, mIoU: {miou:.4f}, Dice: {dice:.4f}")

print("\n--- Test: AttentionVNet ---")
att_vnet = AttentionVNet(in_ch=1, out_ch=3)
lit_att_model = LitBaseVNet(model_obj=att_vnet, in_ch=1, out_ch=3, lr=1e-3)

loss_att, miou_att, dice_att = lit_att_model.compute_loss_and_metrics(batch)
print(f"Loss: {loss_att.item():.4f}, mIoU: {miou_att:.4f}, Dice: {dice_att:.4f}")