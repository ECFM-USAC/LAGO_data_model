import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from datetime import datetime

class LagFileReader:
    def __init__(self, file_path, chunk_size=1000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def _parse_stream(self):
        current_tbp = None
        current_readings = []

        with open(self.file_path, 'r') as file:
            for line in tqdm(file, desc="Processing file"):
                if line.startswith("TBP:"):
                    if current_tbp is not None:
                        yield {'TBP': current_tbp, 'readings': current_readings}
                    current_tbp = int(line.strip().split(":")[1])
                    current_readings = []
                else:
                    stripped = line.strip()
                    if stripped:
                        current_readings.append(float(stripped))

            if current_tbp is not None:
                yield {'TBP': current_tbp, 'readings': current_readings}

    def save_as_parquet_streaming(self, output_folder=".", prefix="instrument_readings"):
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_path = f"{output_folder}/{prefix}__{timestamp}.parquet"
        
        schema = pa.schema([
            ('TBP', pa.int64()),
            ('readings', pa.list_(pa.float64()))
        ])
        
        writer = None
        chunk_data = []
        
        try:
            for record in self._parse_stream():
                chunk_data.append(record)
                
                if len(chunk_data) >= self.chunk_size:
                    table = pa.table(chunk_data, schema=schema)
                    
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, schema, compression='snappy')
                    
                    writer.write_table(table)
                    chunk_data = []
            
            # Write remaining data
            if chunk_data:
                table = pa.table(chunk_data, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression='snappy')
                writer.write_table(table)
                
        finally:
            if writer:
                writer.close()
        
        return output_path