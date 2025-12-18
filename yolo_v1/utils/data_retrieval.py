import os
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = "/n/fs/jborz/projects/casi"

sets = [
    ("2012", "train"),
    ("2012", "val"),
    ("2007", "train"),
    ("2007", "val"),
    ("2007", "test"),
]

print("Downloading Pascal VOC from Kaggle")
root = kagglehub.dataset_download("vijayabhaskar96/pascal-voc-2007-and-2012")
root = os.path.join(root, "VOCdevkit")
print("Dataset downloaded to:", root)

vocdevkit_candidates = [root]
if vocdevkit_candidates:
    voc_root = vocdevkit_candidates[0]
else:
    voc_root = os.path.join(root, "VOCdevkit")
