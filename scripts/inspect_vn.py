import sqlite3

con = sqlite3.connect('data/crawl/runs/run-vn/crawl_manifest.sqlite')
print("Status breakdown:")
for row in con.execute("SELECT status, count(doc_id) FROM crawl_tasks GROUP BY status"):
    print(" ", row)

print("\nFETCHING breakdown by domain:")
for row in con.execute("SELECT domain, count(doc_id) FROM crawl_tasks WHERE status='FETCHING' GROUP BY domain"):
    print(" ", row)

print("\nSUCCESS count in shard_items:")
try:
    for row in con.execute("SELECT count(doc_id) FROM shard_items"):
        print(" ", row)
except Exception as e:
    print(" ", e)
