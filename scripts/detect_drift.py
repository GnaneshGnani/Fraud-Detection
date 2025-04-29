import os
import datetime
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ref_df = pd.read_csv("data/baseline.csv").drop(columns = ["id"])
cur_df = pd.read_csv("data/current_batch.csv").drop(columns = ["id"])

report = Report(metrics = [DataDriftPreset()])
result = report.run(reference_data = ref_df, current_data = cur_df)

drift_score = result.dict()['metrics'][0]['value']['share']

today_date = datetime.datetime.today().strftime('%Y-%m-%d')
result.save_json(f"reports/drift/{today_date}.json")

print("Detected Drift:", drift_score)
if drift_score > 0.5:
    os.environ["GITHUB_OUTPUT"] = "true"
else:
    os.environ["GITHUB_OUTPUT"] = "false"