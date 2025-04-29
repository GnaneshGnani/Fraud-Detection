import glob
import pandas as pd


files = glob.glob("data/incoming/*.csv")

concatenated_files = pd.concat(map(pd.read_csv, files))
concatenated_files.to_csv("data/current_batch.csv", index = False)

print("Merged Incoming Files")