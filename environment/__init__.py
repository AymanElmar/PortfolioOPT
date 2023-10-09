import pandas as pd
from gym.envs.registration import register
from .portfolio import PortfolioEnv

# register our enviroment with combinations of input arguments
df = pd.read_hdf('Poloniex-30m-2023-07-21.h5')

env_specs_args = [

    dict(id='CryptoPortfolioEIIE-v0',
         entry_point='environment.portfolio:PortfolioEnv',
         kwargs=dict(
             steps=3000,
             output_mode='EIIE',
             df=df
         )
         )]
env_specs = [spec['id'] for spec in env_specs_args]

# register our env's on import
for env_spec_args in env_specs_args:
    register(**env_spec_args)