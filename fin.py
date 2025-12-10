import duckdb


db_features: tuple = ('open', 'high', 'low', 'close',
                      'delta_30min', 'ret_30min', 'ret_1day', 'ret_2day',
                      'volume')
select_cols = ',\n  '.join(f'd.{f}' for f in db_features)

cutoff = int(1.703169186e+18)

con = duckdb.connect('/dev/shm/pipeline.duckdb')
con.execute(f"SET memory_limit='300GB'")

def build1(split='train'):
    print(f"  Building {split}_data...")
    con.execute(f"DROP VIEW {split}_data")
    con.execute(f"""
        CREATE TABLE {split}_data AS
        SELECT
            CAST(SUBSTR(d.id, 1, 6) AS INTEGER) AS stock_id,
            ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY d.datetime) - 1 AS row_idx,
            {select_cols}
        FROM df_materialized d
        WHERE epoch_ns(d.datetime) {'<=' if split == 'train' else '>'} {cutoff}
    """)


build1('train')
build1('val')

# Free up memory
con.execute("DROP TABLE df_materialized")

print("Step 7: Creating indexes...")
con.execute("CREATE INDEX train_data_idx ON train_data(stock_id, row_idx)")
con.execute("CREATE INDEX val_data_idx ON val_data(stock_id, row_idx)")

con.close()
exit(0)
print("Step 8: Compute and store quantiles from train_data...")
sample_size = 1000000
self._compute_and_store_quantiles(con, sample_size)

print("Step 9: Build index tables for dataset iteration...")
seq_len = self.seq_len
max_horizon = max(self.horizons)

def build2(split='train'):
    print(f"  Building {split}_index...")
    con.execute(f"""
        CREATE TABLE {split}_index AS
        WITH stock_lengths AS (
            SELECT stock_id, MAX(row_idx) + 1 AS stock_len
            FROM {split}_data
            GROUP BY stock_id
        ),
        valid_stocks AS (
            SELECT
                stock_id,
                stock_len - {seq_len} - {max_horizon} AS valid_samples
            FROM stock_lengths
        )
        SELECT
            stock_id,
            SUM(valid_samples) OVER (ORDER BY stock_id) AS cumsum
        FROM valid_stocks
        WHERE valid_samples > 0
    """)
build2('train')
build2('val')