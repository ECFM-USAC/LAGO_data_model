import pandas as pd
from tqdm import tqdm
from datetime import datetime

class LagFileReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = []

    def parse(self):
        current_tbp = None
        current_readings = []

        with open(self.file_path, 'r') as file:
            for line in tqdm(file, desc="Processing file"):
                if line.startswith("TBP:"):
                    if current_tbp is not None:
                        self.data.append({'TBP': current_tbp, 'readings': current_readings})
                    current_tbp = int(line.strip().split(":")[1])
                    current_readings = []
                else:
                    stripped = line.strip()
                    if stripped:  # avoid empty lines
                        current_readings.append(float(stripped))

            # Add the last section
            if current_tbp is not None:
                self.data.append({'TBP': current_tbp, 'readings': current_readings})

        return self.data

    def to_dataframe(self):
        if not self.data:
            self.parse()
        return pd.DataFrame(self.data)

    def save_as_parquet(self, output_folder: str = ".", prefix: str = "instrument_readings"):
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_path = f"{output_folder}/{prefix}__{timestamp}.parquet"

        df = self.to_dataframe()
        df.to_parquet(output_path, compression='snappy')
        return output_path