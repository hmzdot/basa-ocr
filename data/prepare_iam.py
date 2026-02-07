import kagglehub
import shutil

DATA_DIR = "./data/iam_handwriting"

path = kagglehub.dataset_download("nibinv23/iam-handwriting-word-database")
print(f"Downloaded IAM dataset to {path}")

shutil.move(path, DATA_DIR)
print(f"Moved dataset to {DATA_DIR}")
