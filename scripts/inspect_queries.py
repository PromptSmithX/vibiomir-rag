import pyarrow.parquet as pq

tbl = pq.read_table('data/source/query.parquet')
print(f"Total queries: {len(tbl)}")
for i in range(5):
    qid = tbl['id'][i].as_py()
    text = tbl['query'][i].as_py()
    print(f"[{qid}] {text}")
