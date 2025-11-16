import polars as pl
from datetime import timedelta

def process_chunk(df_chunk: pl.LazyFrame) -> pl.LazyFrame:
    """
    Process a chunk of data grouped by id using vectorized operations.
    Returns a LazyFrame for memory efficiency.
    """
    # Calculate base frequency (mode of time differences) per group
    # Add segment ids to split on large gaps (>31 days)
    return df_chunk.with_columns([
        # Calculate time differences and price deltas
        (pl.col("datetime").diff().dt.total_seconds().alias("duration")),
        (pl.col("close").diff().alias("delta")),
        
        # Create segment ids for gaps > 31 days (split points)
        (pl.col("datetime").diff().dt.total_days() >= 31)
        .cast(int)
        .cum_sum()
        .alias("segment_id")
    ]).with_columns([
        # Calculate base frequency as the most common non-zero duration
        pl.col("duration")
        .filter(pl.col("duration") > 0)
        .mode()
        .first()
        .over("id")
        .alias("base_freq")
    ]).with_columns([
        # Create gap markers using vectorized when/then
        pl.when(pl.col("duration") == 0)
        .then(pl.lit(None))
        .when(pl.col("duration") < pl.col("base_freq"))
        .then(pl.lit("SKIP"))
        .when(pl.col("duration") < timedelta(hours=1).total_seconds())
        .then(pl.lit("SHORT_GAP"))
        .when(pl.col("duration") < timedelta(hours=4).total_seconds())
        .then(pl.lit("LUNCH"))
        .when(pl.col("duration") < timedelta(days=1).total_seconds())
        .then(pl.lit("DINNER"))
        .when(pl.col("duration") < timedelta(days=3).total_seconds())
        .then(pl.lit("WEEKEND"))
        .when(pl.col("duration") < timedelta(days=8).total_seconds())
        .then(pl.lit("SHORT_HOLIDAY"))
        .when(pl.col("duration") < timedelta(days=31).total_seconds())
        .then(pl.lit("HOLIDAY"))
        .otherwise(pl.lit("SPLIT"))
        .alias("gap_type")
    ])

def aggregate_segments(processed: pl.LazyFrame) -> pl.LazyFrame:
    """
    Aggregate the processed data into segments.
    """
    return processed.group_by(["id", "segment_id"]).agg([
        pl.col("datetime").first().alias("start_datetime"),
        pl.col("close").first().alias("start_price"),
        pl.struct([
            pl.col("delta").fill_null(0).alias("deltas").list,
            pl.col("gap_type").alias("markers").list,
            pl.col("duration").sum().alias("total_seconds")
        ]).alias("segment_data")
    ])

def process_all_chunks(df_s: pl.LazyFrame) -> pl.DataFrame:
    """
    Main function to process large DataFrame with multiple ids.
    Leverages all 256 cores automatically.
    """
    # Process all groups in parallel
    processed = process_chunk(df_s)
    
    # Aggregate into segments
    aggregated = aggregate_segments(processed)
    
    # Collect results efficiently
    return aggregated.collect(
        streaming=True,  # Memory-efficient streaming mode
        no_optimization=False
    )

# For ultra-large datasets that don't fit memory, use scan_parquet:
def process_large_parquet(path: str) -> pl.DataFrame:
    """
    Process large parquet files that don't fit in memory.
    """
    lf = pl.scan_parquet(path)
    return process_all_chunks(lf)

if __name__ == "__main__":
    df = pl.scan_parquet('/dev/shm/001000_001500.pq')
    result = process_all_chunks(df)
    breakpoint()
