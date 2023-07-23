import argparse
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=Optional[str], default=None)
args = parser.parse_args()

df = pd.read_json(args.input)

# Since pyarrow has a bug when loading huge json files, we use chunking to convert the json file to dataset
# @see: https://issues.apache.org/jira/browse/ARROW-17137
chunk_size = 500_000
table = pa.Table.from_batches(
    [
        pa.record_batch(df.iloc[i : i + chunk_size])
        for i in range(0, len(df), chunk_size)
    ]
)
pq.write_table(
    table,
    args.output if args.output is not None else args.input.replace(".json", ".parquet"),
)

print("Successfully converted.")
