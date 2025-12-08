#!/usr/bin/env python3
"""
scripts/extract_range.py

Usage:
  python scripts/extract_range.py --start '2017-01-01' --end '2017-03-01' \
    --in data/poloniex_30m.hf --out data/poloniex_30m_subset.hf

This script reads the project's HDF dataset and writes a new HDF containing
'train' and 'test' keys where 'test' is the requested datetime window and
'train' is all data strictly before the test start.

It keeps the same structure so `compare_strategies.py` can be pointed at the
new file with the `DATA_PATH` env var.
"""
import argparse
import pandas as pd
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", default="data/poloniex_30m.hf")
    p.add_argument("--out", dest="outfile", default=None)
    p.add_argument("--start", required=True,
                   help="Test start timestamp (inclusive), e.g. '2017-01-01 00:00:00'")
    p.add_argument("--end", required=True,
                   help="Test end timestamp (inclusive), e.g. '2017-03-01 00:00:00'")
    return p.parse_args()


def main():
    args = parse_args()
    infile = args.infile
    test_start = pd.to_datetime(args.start)
    test_end = pd.to_datetime(args.end)
    outfile = args.outfile or infile.replace('.hf', f'_test_{test_start.date()}_{test_end.date()}.hf')

    if not os.path.exists(infile):
        raise SystemExit(f"Input file not found: {infile}")

    # Load the original HDF
    df_train_full = pd.read_hdf(infile, key='train')
    df_test_orig = pd.read_hdf(infile, key='test')

    # Some stored DatetimeIndex objects include a non-standard 'freq' bytes
    # value which can cause pandas.concat to fail. Normalize indices to
    # plain DatetimeIndex without freq before concatenation.
    df_train_full.index = pd.DatetimeIndex(df_train_full.index.values)
    df_test_orig.index = pd.DatetimeIndex(df_test_orig.index.values)

    # Combine and sort (some rows for the requested range may be in either key)
    df_all = pd.concat([df_train_full, df_test_orig], verify_integrity=False)
    df_all = df_all.sort_index()

    # Select test slice
    df_test = df_all.loc[(df_all.index >= test_start) & (df_all.index <= test_end)]
    if df_test.empty:
        raise SystemExit("No rows found in the requested test range. Check timestamps and timezone.")

    # Train is everything before test_start
    df_train_new = df_all.loc[df_all.index < test_start]

    # Write to HDF
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with pd.HDFStore(outfile, 'w') as store:
        store.put('train', df_train_new)
        store.put('test', df_test)

    print(f"Wrote {outfile}: train={len(df_train_new)} rows, test={len(df_test)} rows")


if __name__ == '__main__':
    main()
