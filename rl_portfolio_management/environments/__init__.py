import pandas as pd
import os
from gymnasium.envs.registration import register
from .portfolio import PortfolioEnv

# register our enviroment with combinations of input arguments
# Use absolute path relative to this file
_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
df_train = pd.read_hdf(os.path.join(_data_dir, 'poloniex_30m.hf'), key='train')

env_specs_args = [
    dict(id='CryptoPortfolioMLP-v0',
         entry_point='rl_portfolio_management.environments.portfolio:PortfolioEnv',
         kwargs=dict(
             output_mode='mlp',
             df=df_train
         )),
    dict(id='CryptoPortfolioEIIE-v0',
         entry_point='rl_portfolio_management.environments.portfolio:PortfolioEnv',
         kwargs=dict(
             output_mode='EIIE',
             df=df_train
         )
         ),
    dict(id='CryptoPortfolioAtari-v0',
         entry_point='rl_portfolio_management.environments.portfolio:PortfolioEnv',
         kwargs=dict(
             output_mode='atari',
             df=df_train
         ))
]
env_specs = [spec['id'] for spec in env_specs_args]

# register our env's on import
for env_spec_args in env_specs_args:
    register(**env_spec_args)
