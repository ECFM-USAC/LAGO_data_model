import sys
import os
from lago_data_model.reader import LagFileReader

if len(sys.argv) < 2:
    print("Please provide the path to a .lag file.")
    print("Usage: poetry run python scripts/run_lag_reader.py path/to/file.lag")
    sys.exit(1)

file_path = sys.argv[1]

reader = LagFileReader(file_path)
reader.parse()

file_name = os.path.splitext(os.path.basename(file_path))[0]

output = reader.save_as_parquet("./data", prefix=file_name)

print(f"Saved Parquet to: {output}")