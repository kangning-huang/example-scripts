import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from io_cmip6 import read_cmip6_csv
from run_teb import run_teb
from compute_G import compute_G


def load_cfg():
    with open('config/canyon.yml') as f:
        return yaml.safe_load(f)


def test_pipeline_runs(tmp_path):
    df = read_cmip6_csv('data/mock_cmip6.csv')
    cfg = load_cfg()
    teb_df = run_teb(df, cfg)
    result = compute_G(teb_df, cfg)
    out_file = tmp_path / 'g_hourly.csv'
    result.to_csv(out_file, index=False)
    assert out_file.exists()


def test_sun_shade_difference():
    df = read_cmip6_csv('data/mock_cmip6.csv').iloc[[0]]
    cfg = load_cfg()
    cfg_shade = cfg.copy()
    cfg_shade['human'] = cfg['human'].copy()
    cfg_sun = cfg.copy()
    cfg_sun['human'] = cfg['human'].copy()
    cfg_sun['human']['scenario'] = 'sun'
    shade = compute_G(run_teb(df, cfg_shade), cfg_shade)
    sun = compute_G(run_teb(df, cfg_sun), cfg_sun)
    assert sun['Rin'].iloc[0] > shade['Rin'].iloc[0]
    assert sun['G_Wm2'].iloc[0] >= shade['G_Wm2'].iloc[0]


def test_lambdaE_max_effect():
    df = read_cmip6_csv('data/mock_cmip6.csv')
    cfg = load_cfg()
    high = compute_G(run_teb(df, cfg), cfg)
    cfg_low = cfg.copy()
    cfg_low['human'] = cfg['human'].copy()
    cfg_low['human']['lambdaE_max_Wm2'] = 300
    low = compute_G(run_teb(df, cfg_low), cfg_low)
    assert (low['G_Wm2'] >= high['G_Wm2']).all()
    assert (low['uncompensable'].astype(int) >= high['uncompensable'].astype(int)).any()
