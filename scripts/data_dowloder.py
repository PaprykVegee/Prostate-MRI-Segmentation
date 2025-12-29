import kagglehub
import os
import pathlib
import shutil

download_path = pathlib.Path("Prostate-MRI-Segmentation/data")
os.makedirs(download_path, exist_ok=True)

path = pathlib.Path(kagglehub.dataset_download("vamshivardhanemmadi/prostate158"))

print("Pobrano do:", path)

for item in path.iterdir():
    shutil.move(str(item), download_path)

print("Dataset PRZENIESIONY do:", download_path)
