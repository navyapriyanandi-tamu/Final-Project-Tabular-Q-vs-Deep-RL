#!/usr/bin/env python3
"""
scripts/run_ranges.py

Generate sliding test windows, extract dataset subsets, and run
`compare_strategies.py` for each subset. Logs for each run are stored in
`results/runs/<start>_<end>.log`.

By default this script will:
 - create 3-month test windows
 - step by 1 month
 - run compare_strategies.py for each window (can be expensive)

Usage examples:
  python scripts/run_ranges.py --window-months 3 --step-months 1 --start '2015-01-01' --end '2017-06-01'

Note: runs `compare_strategies.py` in a subprocess; ensure you're in the
project virtualenv when running this script.
"""
import argparse
import subprocess
import os
import sys
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--start', required=True, help='Earliest test window start (inclusive)')
    p.add_argument('--end', required=True, help='Latest test window end (inclusive)')
    p.add_argument('--window-months', type=int, default=3)
    p.add_argument('--step-months', type=int, default=1)
    p.add_argument('--data-in', default='data/poloniex_30m.hf')
    p.add_argument('--out-dir', default='results/runs')
    p.add_argument('--dry-run', action='store_true', help='Only list windows, do not run')
    return p.parse_args()


def windows(start_dt, end_dt, window_months=3, step_months=1):
    cur = start_dt
    while True:
        w_end = cur + relativedelta(months=window_months) - pd.Timedelta(minutes=30)
        if w_end > end_dt:
            break
        yield cur, w_end
        cur = cur + relativedelta(months=step_months)


def classify_windows(infile, window_list, bull_thresh=0.05, neut_thresh=0.0):
    """Classify each window as bullish/neutral/bearish using equal-weight buy&hold.

    Returns dict mapping (start,end) -> class_str
    """
    # Load and prepare close prices
    df_train = pd.read_hdf(infile, key='train')
    df_test = pd.read_hdf(infile, key='test')
    for d in (df_train, df_test):
        d.index = pd.DatetimeIndex(d.index.values)
    df = pd.concat([df_train, df_test]).sort_index()
    closes = df.xs('close', axis=1, level=1)

    cls_map = {}
    for s, e in window_list:
        sub = closes.loc[(closes.index >= s) & (closes.index <= e)]
        if sub.empty:
            cls_map[(s, e)] = 'empty'
            continue
        sub = sub.dropna(axis=1, how='any')
        if sub.shape[1] == 0:
            cls_map[(s, e)] = 'empty'
            continue
        p0 = sub.iloc[0]
        p_end = sub.iloc[-1]
        rel = (p_end / p0)
        port_ratio = rel.mean()
        cum_return = port_ratio - 1.0
        if cum_return > bull_thresh:
            cls = 'bullish'
        elif cum_return > neut_thresh:
            cls = 'neutral'
        else:
            cls = 'bearish'
        cls_map[(s, e)] = cls
    return cls_map


def run_one(start_ts, end_ts, infile, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subset_name = f"data/poloniex_30m_test_{start_ts.date()}_{end_ts.date()}.hf"

    # call extract_range
    cmd = [sys.executable, 'scripts/extract_range.py', '--in', infile, '--out', subset_name, '--start', str(start_ts), '--end', str(end_ts)]
    print('Extracting:', ' '.join(cmd))
    r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)

    # Run compare_strategies.py with DATA_PATH set
    # logfile will be suffixed with the market class if available
    # allow passing WINDOW_CLASS env var from caller
    cls = os.environ.get('WINDOW_CLASS') or 'unknown'
    logfile = os.path.join(out_dir, f"run_{start_ts.date()}_{end_ts.date()}_{cls}.log")
    env = os.environ.copy()
    env['DATA_PATH'] = subset_name
    run_cmd = [sys.executable, 'compare_strategies.py']
    print('Running:', ' '.join(run_cmd), '->', logfile)
    with open(logfile, 'w') as fh:
        proc = subprocess.Popen(run_cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        proc.wait()
    print('Finished, log ->', logfile)


def main():
    args = parse_args()
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)

    to_run = list(windows(start_dt, end_dt, args.window_months, args.step_months))
    print(f'Will run {len(to_run)} windows from {start_dt} to {end_dt}')

    # Classify windows up-front so we can include the class in filenames
    cls_map = classify_windows(args.data_in, to_run)

    for s, e in to_run:
        cls = cls_map.get((s, e), 'unknown')
        print('Window:', s, '->', e, '| class:', cls)
        if args.dry_run:
            continue
        try:
            # pass class to run_one via environment variable so log naming includes it
            env = os.environ.copy()
            env['WINDOW_CLASS'] = cls
            # call extract_range subprocess + compare_strategies as before
            subset_name = f"data/poloniex_30m_test_{s.date()}_{e.date()}.hf"
            cmd_extract = [sys.executable, 'scripts/extract_range.py', '--in', args.data_in, '--out', subset_name, '--start', str(s), '--end', str(e)]
            print('Extracting:', ' '.join(cmd_extract))
            subprocess.run(cmd_extract, check=True, env=env)

            logfile = os.path.join(args.out_dir, f"run_{s.date()}_{e.date()}_{cls}.log")
            run_cmd = [sys.executable, 'compare_strategies.py']
            print('Running:', ' '.join(run_cmd), '->', logfile)
            with open(logfile, 'w') as fh:
                proc = subprocess.Popen(run_cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
                proc.wait()
            print('Finished, log ->', logfile)
        except subprocess.CalledProcessError as ex:
            print('Error running window', s, e, ex)


if __name__ == '__main__':
    main()
