from pathlib import Path
from dataloader import create_dataloaders

root_dir = str((Path.cwd() / "data" / "msd").resolve())

train_loader, val_loader, test_loader, meta = create_dataloaders(
    root_dir=root_dir,
    spacing=(1.5, 1.5, 2.0),
    roi_size=(128, 128, 64),
    batch_size=2,
    num_workers=2,
    val_frac=0.2,
    augment=True,
    cache_dataset=False,
    seed=42,
)

print("Data root:", root_dir)
