#!/usr/bin/env python3
"""
scripts/make_dummy_tests.py

Create three modified versions of the original test set to simulate bullish,
bearish and neutral market regimes and write new HDF files that keep the
original `train` and `val` splits intact. The new files will be written as:

 - data/poloniex_30m_test_dummy_bullish.hf
 - data/poloniex_30m_test_dummy_bearish.hf
 - data/poloniex_30m_test_dummy_neutral.hf

The script multiplies all price columns by a smoothly varying multiplicative
factor to induce an overall trend while keeping the data structure the same.
"""
import os
import numpy as np
import pandas as pd

INFILE = 'data/poloniex_30m.hf'
OUT_DIR = 'data'


def make_trend_factors(index, kind='bullish', magnitude=0.5):
    """Return multiplicative factors for each timestamp.

    - bullish: linear ramp from 1 -> 1+magnitude
    - bearish: linear ramp from 1 -> 1-magnitude
    - neutral: random small noise around 1 (mean 1)
    """
    n = len(index)
    if kind == 'bullish':
        return 1.0 + np.linspace(0.0, magnitude, n)
    elif kind == 'bearish':
        # ensure factors remain positive
        return 1.0 - np.linspace(0.0, magnitude, n)
    elif kind == 'neutral':
        # small Gaussian fluctuations around 1, cumulative to avoid huge jumps
        rng = np.random.default_rng(42)
        steps = rng.normal(loc=0.0, scale=0.005, size=n)  # ~0.5% noise per step
        # cumulative multiplicative effects
        fac = np.cumprod(1.0 + steps)
        # normalize to have mean ~1
        fac = fac / np.mean(fac)
        return fac
    else:
        raise ValueError('unknown kind')


def apply_factors_to_prices(df_test, factors):
    """Multiply all numeric price columns by factors (one factor per row).

    The dataset uses a MultiIndex columns (asset, price_type). We'll multiply
    all prices (open/high/low/close) consistently.
    """
    df = df_test.copy()
    # ensure index alignment
    assert len(df) == len(factors)
    # Multiply every column by the factor per row
    # handle numeric dtypes only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # broadcast multiplication
    df[numeric_cols] = (df[numeric_cols].values.T * factors).T
    return df


def make_dummy_files(infile=INFILE, out_dir=OUT_DIR):
    if not os.path.exists(infile):
        raise SystemExit(f'Input file not found: {infile}')

    df_train = pd.read_hdf(infile, key='train')
    df_test = pd.read_hdf(infile, key='test')

    # normalize indices
    df_train.index = pd.DatetimeIndex(df_train.index.values)
    df_test.index = pd.DatetimeIndex(df_test.index.values)

    for kind in ['bullish', 'neutral', 'bearish']:
        print('Creating dummy test:', kind)
        factors = make_trend_factors(df_test.index, kind=kind, magnitude=0.5)
        df_test_new = apply_factors_to_prices(df_test, factors)
        outfile = os.path.join(out_dir, f'poloniex_30m_test_dummy_{kind}.hf')
        with pd.HDFStore(outfile, 'w') as store:
            store.put('train', df_train)
            store.put('test', df_test_new)
        print('Wrote', outfile, 'train=', len(df_train), 'test=', len(df_test_new))


if __name__ == '__main__':
    make_dummy_files()
