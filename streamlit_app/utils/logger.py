import os
import csv
from datetime import datetime

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "data",
    "detection_log.csv"
)

def log_prediction(model, source, text, prediction, confidence):

    file_exists = os.path.isfile(LOG_FILE)

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        # Write header only once
        if (not file_exists) or os.path.getsize(LOG_FILE) == 0:

            writer.writerow([
                "Timestamp",
                "Model",
                "Source",
                "Text",
                "Prediction",
                "Confidence"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model,
            source,
            text,
            prediction,
            round(confidence * 100, 2)
        ])