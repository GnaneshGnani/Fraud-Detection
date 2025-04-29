import os

os.remove("data/baseline.csv")
os.rename("data/current_batch.csv", "data/baseline.csv")
for file in os.listdir("data/incoming"):
    os.remove(f"data/incoming/{file}")