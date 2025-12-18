import duckdb
import h5py
import numpy as np
import polars as pl
from glob import glob
from pathlib import Path

pattern = '/data/share/data/*/*_mrq_0.h5'


def load_h5_to_duckdb(pattern: str, db_path: str = ':memory:') -> duckdb.DuckDBPyConnection:
    """
    Read every file matching the pattern into DuckDB.
    Folder name becomes table name, filename (without _mrq_0.h5) becomes column name.
    All .h5 files have the same shape: ticker, time, and values (time x ticker matrix).
    """
    files = glob(pattern)
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    # Group files by folder (table name)
    tables: dict[str, list[Path]] = {}
    for f in files:
        p = Path(f)
        table_name = p.parent.name
        tables.setdefault(table_name, []).append(p)

    con = duckdb.connect(db_path)

    for table_name, file_list in tables.items():
        # Load first file to get ticker and time dimensions
        with h5py.File(file_list[0], 'r') as h5f:
            tickers = [t.decode() for t in h5f['ticker'][:]]
            times = h5f['time'][:]

        n_times = len(times)
        n_tickers = len(tickers)

        # Build data dict for polars
        data = {
            'time': np.repeat(times, n_tickers),
            'ticker': np.tile(tickers, n_times),
        }

        # Add each file as a column
        for file_path in file_list:
            col_name = file_path.stem.replace('_mrq_0', '')
            with h5py.File(file_path, 'r') as h5f:
                values = h5f['values'][:]  # shape: (time, ticker)
                data[col_name] = values.flatten()

        df = pl.DataFrame(data).with_columns(
            pl.col('time').cast(pl.Datetime('ns'))
        )

        # Register as table
        con.register(f'{table_name}_df', df)
        con.execute(f'CREATE TABLE {table_name} AS SELECT * FROM {table_name}_df')
        con.unregister(f'{table_name}_df')
        print(f"Created table '{table_name}' with {len(file_list)} columns, {len(df)} rows")

    return con


if __name__ == '__main__':
    con = load_h5_to_duckdb(pattern)
    # Show tables and sample data
    print("\nTables:")
    print(con.execute("SHOW TABLES").fetchall())
    for table in con.execute("SHOW TABLES").fetchall():
        table_name = table[0]
        print(f"\n{table_name} columns:")
        print(con.execute(f"DESCRIBE {table_name}").fetchall()[:5])
