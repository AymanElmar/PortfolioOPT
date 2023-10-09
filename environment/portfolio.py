import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pprint as pprint # pretty print 
import logging # log messages
import os 
import tempfile # temporary files
import time
import gym # openai gym
import gym.spaces # gym spaces
import datetime # datetime 
import sys # system calls
import math # math functions

from util import MDD as max_drawdown, sharpe, softmax # maximum drawdown sharpe ratio and softmax
from callbacks.notebook_plot import LivePlotNotebook

eps=1e-8 # epsilon constant

# Create a file handler and set its level to DEBUG
ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
save_path = './outputs/PG/PGportfolio_crypto-%s' % ts
log_dir = os.path.join('logs', os.path.splitext(os.path.basename(save_path))[0], 'run-' + ts)
try:
    os.makedirs(log_dir)
except OSError:
    pass
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False


# Remove any existing console handlers from the logger
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        logger.removeHandler(handler)

# Create a file handler and set its level to DEBUG
handler = logging.FileHandler(log_dir+'/logfile.log')
handler.setLevel(logging.DEBUG)

# Create a formatter and set it on the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)




class DataSrc(object):
    """This class is used by the Environment class to get the data for each new episode. It is not used directly by the agent."""
    def __init__(self, df, steps=252, scale=True, scale_extra_cols=True, augment=0.00, window_length=50, random_reset=True,prevent_overfitting=False):
        """
        DataSrc is a class that provides the data for each new episode. 
        It is used by the Environment class to get the data for each new episode. 
        It is not used directly by the agent.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data for the episode.
            csv for data frame index of timestamps and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close',...]]
             an example is included as an hdf file in this repository
        steps : int
            Number of steps for the episode.
        scale : bool
            Whether to scale the data or not.
        scale_extra_cols : bool
            Whether to scale extra columns or not.
        augment : float
            Noise to add to the data.
        window_length : int
            Size of the observation window.
        start_idx : int
            Episode starting index.
        random_reset : bool
            Whether to reset randomly or not.
        """
        self.df = df
        self.steps = steps+ 1
        self.scale = scale
        self.scale_extra_cols = scale_extra_cols
        self.augment = augment
        self.window_length = window_length
        self.idx = self.window_length
        self.random_reset = random_reset
        self.prevent_overfitting = prevent_overfitting
              # get rid of NaN's
        df = df.copy()
        df.replace(np.nan, 0, inplace=True)
        df = df.fillna(method="pad")

        # dataframe to matrix
        self.asset_names = df.columns.levels[0].tolist()
        self.features = df.columns.levels[1].tolist()
        self.price_columns = ['close', 'high', 'low']
        data = df.to_numpy().reshape(
            (len(df), len(self.asset_names), len(self.price_columns)))
        self._data = np.transpose(data, (1, 0, 2)).copy()
        self._times = df.index

        self.non_price_columns = set(
            df.columns.levels[1]) - set(self.price_columns)
        
        # Stats to let us normalize non price columns
        # To be added 

        self.reset()

    def _step(self):
        """this function is called at each step of the episode it takes the data for the current step and returns the history matrix, the price relative vector and a boolean indicating if the episode is done or not"""
        # get history matrix from dataframe

        data_window = self.data[:, self.step:self.step +
                                self.window_length].copy()
        print("step: ", self.step ) if self.step%500==0 else None
        # (eq.1) prices
        y1 = data_window[:, -1, 0] / (data_window[:, -2, 0]+ eps)
        y1=np.nan_to_num(y1)
        y1=np.insert(y1,0, 1.0)

        # (eq 18) X: prices are divided by close price
        nb_pc = len(self.price_columns)
        if self.scale:
            last_price_vect = data_window[:, -1, :]
            data_window[:, :, :] /= last_price_vect[:,
                                                          np.newaxis, :]+ eps
        data_window=np.nan_to_num(data_window)


        self.step += 1
        history = data_window
        history= history.astype(np.float32)
        done = bool(self.step >= self.steps)

        return history, y1, done
    
    def reset(self):
        """this function is called at the beginning of each episode it resets the data and returns the history matrix, the price relative vector and a boolean indicating if the episode is done or not"""
        self.step = 0
        print("reseting data")
        # get data for this episode
        if self.random_reset:
            self.idx = np.random.randint(
                low=self.window_length + 1, high=self._data.shape[1] - self.steps - 2)
        else:
            # continue sequentially, before reseting to start
            if self.idx>(self._data.shape[1] - self.steps - self.window_length - 1):
                self.idx=self.window_length + 1
            else:
                self.idx += self.steps
        data = self._data[:, self.idx -
                          self.window_length:self.idx + self.steps + 1].copy()
        self.times = self._times[self.idx -
                                 self.window_length:self.idx + self.steps + 1]

        if self.prevent_overfitting:   
        # augment data to prevent overfitting
            data += np.random.normal(loc=0, scale=self.augment, size=data.shape)
        self.data = data
        print("idx: ", self.idx)



class PortfolioSim(object):
    """this class is used by the Environment class to simulate the portfolio. It is not used directly by the agent."""

    def __init__(self, asset_names=[], steps=128, trading_cost=0.0025, time_cost=0.0):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names
        self.reset()

    def _step(self, w1, y1):
        """
        Step.
        it takes the portfolio weights and the price relative vector and returns the reward, the info dictionary and a boolean indicating if the episode is done or not
        
        w1 - new action of portfolio weights - e.g. [0.1,0.9, 0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        """
        w0 = self.w0
        p0 = self.p0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        # (eq16) cost to change portfolio
        # (excluding change in cash to avoid double counting for transaction cost)
        c1 = self.cost * (
            np.abs(dw1[1:] - w1[1:])).sum()

        p1 = p0 * (1 - c1) * np.dot(y1, w0)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        # can't have negative holdings in this model (no shorts)
        p1 = np.clip(p1, 0, np.inf)

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # (eq10) log rate of return
        # (eq22) immediate reward is log rate of return scaled by episode length
        reward = r1 / self.steps

        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done
        done = math.isclose(p1, 0,rel_tol=1e-03, abs_tol=0.0)

        # should only return single values, not list
        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": c1,
        }
        # record weights and prices
        for i, name in enumerate(['Capital'] + self.asset_names):
            info['weight_' + name] = w1[i]
            info['price_' + name] = y1[i]

        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 =np.array( [1.0] + [0.0] * len(self.asset_names))
        self.w0= self.w0.astype(np.float32)
        self.p0 = 1.0

class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.

    """

    metadata = {'render.modes': ['notebook', 'ansi']}

    def __init__(self,
                 df,
                 steps=256,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=16,
                 augment=0.00,
                 output_mode='EIIE',
                 log_dir=log_dir,
                 scale=True,
                 scale_extra_cols=True,
                 random_reset=True
                 ):
        """
        An environment for financial portfolio management.

        Params:
            df - csv for data frame index of timestamps
                 and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close']]
            steps - steps in episode
            window_length - how many past observations["history"] to return
            trading_cost - cost of trade as a fraction,  e.g. 0.0025 corresponding to max rate of 0.25% at Poloniex (2017)
            time_cost - cost of holding as a fraction
            augment - fraction to randomly shift data by
            output_mode: decides observation["history"] shape
            - 'EIIE' for (assets, window, 3)
            - 'atari' for (window, window, 3) (assets is padded)
            - 'mlp' for (assets*window*3)
            log_dir: directory to save plots to
            scale - scales price data by last opening price on each episode (except return)
            scale_extra_cols - scales non price data using mean and std for whole dataset
        """
        self.src = DataSrc(df=df, steps=steps, scale=scale, scale_extra_cols=scale_extra_cols,
                           augment=augment, window_length=window_length,
                           random_reset=random_reset)
        self._plot = self._plot2 = self._plot3 = self._plot4 = None
        self.output_mode = output_mode
        self.sim = PortfolioSim(
            asset_names=self.src.asset_names,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)
        self.log_dir = log_dir

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        nb_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.Box(
            0.0, 1.0, shape=(nb_assets + 1,))

        # get the history space from the data min and max
        if output_mode == 'EIIE':
            obs_shape = (
                nb_assets,
                window_length,
                3
            )
        else:
            raise Exception('Invalid value for output_mode: %s' %
                            self.output_mode)

        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(
                -10,
                20 if scale else 1,  # if scale=True observed price changes return could be large fractions
                obs_shape
            ),
            'weights': self.action_space
        })
        self.reset()

    def step(self, action):
        """
        Step the env.

        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight between 0 and 1. The first (w0) is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        logger.debug('action: %s', action)

        weights = np.clip(action, 0.0, 1.0)
        weights /= weights.sum() + eps

        # Sanity checks
        assert self.action_space.contains(
            action), 'action should be within %r but is %r' % (self.action_space, action)
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        history, y1, done1 = self.src._step()

        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod(
            [inf["market_return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.src.times[self.src.step].timestamp()
        info['steps'] = self.src.step

        self.infos.append(info)

        # reshape history according to output mode
        if self.output_mode == 'EIIE':
            pass
        if done1 or done2:
            #output info file txt
            with open(self.log_dir + '/info.txt', 'w') as f:
                print(self.infos, file=f)
            
        return {'history': history, 'weights': weights}, reward, done1 or done2, info   

    def reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward, done, info = self.step(action)
        return observation

    def _seed(self, seed):
        np.random.seed(seed)
        return [seed]


    def _render(self, mode='notebook', close=False):
        # if close:
            # return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'notebook':
            self.plot_notebook(close)

    def plot_notebook(self, close=False):
        """Live plot using the jupyter notebook rendering of matplotlib."""

        if close:
            self._plot = self._plot2 = self._plot3 = self._plot4=None
            return

        df_info = pd.DataFrame(self.infos)
        df_info.index = pd.to_datetime(df_info["date"], unit='s')

        # plot prices and performance
        all_assets =  self.sim.asset_names
        if not self._plot:
            colors = [None] * len(all_assets) + ['black']
            self._plot_dir = os.path.join(
                self.log_dir, 'notebook_plot_prices_' + str(time.time())) if self.log_dir else None
            self._plot = LivePlotNotebook(
                log_dir=self._plot_dir, title='prices & performance', labels=all_assets + ["Portfolio"], ylabel='value', colors=colors)
        x = df_info.index
        y_portfolio = df_info["portfolio_value"]
        y_assets = [df_info['price_' + name].cumprod()
                    for name in all_assets]
        self._plot.update(x, y_assets + [y_portfolio])


        # plot portfolio weights
        if not self._plot2:
            self._plot_dir2 = os.path.join(
                self.log_dir, 'notebook_plot_weights_' + str(time.time())) if self.log_dir else None
            self._plot2 = LivePlotNotebook(
                log_dir=self._plot_dir2, labels=all_assets, title='weights', ylabel='weight')
        ys = [df_info['weight_' + name] for name in all_assets]
        self._plot2.update(x, ys)

        # plot portfolio costs
        if not self._plot3:
            self._plot_dir3 = os.path.join(
                self.log_dir, 'notebook_plot_cost_' + str(time.time())) if self.log_dir else None
            self._plot3 = LivePlotNotebook(
                log_dir=self._plot_dir3, labels=['cost'], title='costs', ylabel='cost')
        ys = [df_info['cost'].cumsum()]
        self._plot3.update(x, ys)

        if not self._plot4:
            self._plot_dir4 = os.path.join(
                self.log_dir, 'notebook_plot_portfolio_' + str(time.time())) if self.log_dir else None
            self._plot4 = LivePlotNotebook(
                log_dir=self._plot_dir4,labels=['portfolio'], title='portfolio', ylabel='value')
            x= df_info.index
            y= df_info["portfolio_value"]
            self._plot4.update(x, y)
        if close:
            self._plot = self._plot2 = self._plot3 =self._plot4= None