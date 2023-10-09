# Portfolio Optimization using deep Reinforcement Learning.
## Part One: Creating the environment.
### a) creating the data provider
the data provider is a class that will provide the data to the environment. It will be used to get the data from the database, and to preprocess it.
```python
class DataSrc(object):
    """This class is used by the Environment class to get the data for each new episode. It is not used directly by the agent."""
    def __init__(self, df, steps=252, scale=True, scale_extra_cols=True, augment=0.00, window_length=50, random_reset=True,prevent_overfitting=False):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Data for the episode.
            csv for data frame index of timestamps and multi-index columns levels=[['BTC'],...],['open','low','high','close',...]]
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
        random_reset : bool
            Whether to reset randomly or not.
        prevent_overfitting : bool
            Whether to augment data or not.
        """
        self.df = df
        self.steps = steps+ 1
        self.scale = scale
        self.scale_extra_cols = scale_extra_cols
        self.augment = augment
        self.window_length = window_length
        self.idx = self.window_length+1
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
```

the init function takes data and other parameters, transforms it to a matrix and stores it in self._data, then calls for the reset function.


the reset function is called at the beginning of each episode,it takes the data, takes a chunk of it (containing all the necessary data for the entire episode(steps+history)),augments it if enabled and stores it in self.data.


the _step function activates each step, it takes data and returns the y1 vector which is the latest price relative fluctuations, the history matrix and a boolean indicating if the episode is done or not.


![self.data](readme_images\selfdotdata.png) 
>this image shows how history is extracted from self.data at each step

![self.data](readme_images\classdatasrc.png)
>this image shows the inputs and outputs of the class.

### b) creating the portfolio simulator
the portfolio simulator is a class that will simulate the portfolio, it will be used to calculate the reward and the portfolio value.
```python
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
        Numbered equations are from https://arxiv.org/abs/1706.10059
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

```
the init function takes the asset names, the number of steps, the trading cost and the time cost. then calls for the reset function.

the reset function resets the portfolio to the initial state. which is 1 btc and 0 of the other assets.

the _step function takes the new weights I allocated to each assets, checks the price fluctuations and calculates the reward and the portfolio value. if the portfolio value is 0 aka we ran out of money then the episode is done.

![self.data](readme_images\portfoliosim.png)
>this image shows the inputs and outputs of the class portfoliosim

### c) creating the environment
the environment is a class that will be used by the agent to interact with the data provider and the portfolio simulator.
```python
class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.

    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.

    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
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
```
this class extends a gym environment.

the init function takes the data, number of steps,history length and other parameters. then calls for the reset function.

the reset function resets the previous two classes and returns the first observation.

the step function takes the action, checks if it is valid, then calls for the _step function of the data provider, having an observation and an action it then calls for the _step function of the portfolio simulator, then returns the observation, reward, done and info.

##Part Two: Creating the agent.
### a) wrapping the environment
since the agent takes a single input and returns a single output, we need to wrap the environment to make it compatible with the agent.
```python
def concat_states(state):
    history = state["history"]
    weights = state["weights"]
    weight_insert_shape = (history.shape[0], 1, history.shape[2])
    if len(weights) - 1 == history.shape[0]:
        weight_insert = np.ones(
            weight_insert_shape) * weights[1:, np.newaxis, np.newaxis]
    elif len(weights) - 1 == history.shape[2]:
        weight_insert = np.ones(
            weight_insert_shape) * weights[np.newaxis, np.newaxis, 1:]
    else:
        weight_insert = np.ones(
            weight_insert_shape) * weights[np.newaxis, 1:, np.newaxis]
    state = np.concatenate([weight_insert, history], axis=1)
    return state


class ConcatStates(gym.Wrapper):
    """
    Concat both state arrays for models that take a single inputs.

    Usage:
        env = ConcatStates(env)

    """

    def __init__(self, env):
        super().__init__(env)
        hist_space = self.observation_space.spaces["history"]
        hist_shape = hist_space.shape
        self.observation_space = gym.spaces.Box(-10, 10, shape=(
            hist_shape[0], hist_shape[1] + 1, hist_shape[2]))

    def step(self, action):

        state, reward, done, info = self.env.step(action)

        # concat the two state arrays, since some models only take a single output
        state = concat_states(state)

        return state, reward, done, info

    def reset(self):
        self.env.reset()
        state = self.env.reset()
        return concat_states(state)
```
the concat_states function takes the observation and returns a single matrix containing the history and the weights.
### b) creating the policy network for the agent
the policy network is a neural network that will be used by the agent to predict the action.
```python
class DepthCNN(nn.Module):
    def __init__(self, channels, ratio=16 ,bias=True,num_assets=13,hist_len=10,cash_bias_int=0,device='cpu'):
        super(DepthCNN,self).__init__()
        self.channels = channels
        self.convs1x1 = nn.Conv2d(channels, 6, kernel_size=(1, 1), padding=(0, 0))
        self.convs1x5 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2)) for _ in range(0,6)]).to(device)
        self.convs5x1 = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=(5, 1), padding=(2, 0)) for _ in range(0,6)]).to(device)
        self.ratio = ratio
        self.Cdense1 = nn.Linear(12, 12 * ratio, bias=bias).to(device)
        self.Cdense2 = nn.Linear(12 * ratio,12, bias=bias).to(device)
        self.Hdense1 = nn.Linear(num_assets, num_assets * ratio, bias=bias).to(device)
        self.Hdense2 = nn.Linear(num_assets * ratio, num_assets, bias=bias).to(device)
        self.Wdense1 = nn.Linear(hist_len, hist_len * ratio, bias=bias).to(device)
        self.Wdense2 = nn.Linear(hist_len * ratio, hist_len, bias=bias).to(device)
        self.convs1x1two = nn.Conv2d(13, 1, kernel_size=(1, 1), padding=(0, 0)).to(device)
        self.cash_bias_int = cash_bias_int 
        self.device = device
    def forward(self,inputs,ifstates=1):
        cash_bias = torch.tensor(self.cash_bias_int).repeat(ifstates,1,1,1).to(self.device)       
        w0 = inputs[:, :1, :, :1].to(self.device) 
        inputs=inputs[:, :, :, 1:].to(self.device)
        x = self.convs1x1(inputs)
        x = torch.relu(x)
        output_channels = np.split(x,x.shape[1], axis=1)
        out = []
        for i in range(0,6):
            layer = self.convs1x5[i](output_channels[i])
            out.append(layer)
            layer = self.convs5x1[i](output_channels[i])
            out.append(layer)
        concatenated = torch.cat(out, dim=1)
        b, c, h, w = concatenated.shape
        C = torch.mean(concatenated, dim=[2, 3])[0]
        C = self.Cdense1(C)
        C= torch.relu(C)
        C = torch.sigmoid(self.Cdense2(C))
        C = C.view(1,c, 1, 1)
        W= torch.mean(concatenated, dim=[1, 2])[0]
        W = self.Wdense1(W)
        W= torch.relu(W)
        W = torch.sigmoid(self.Wdense2(W))
        W = W.view(1, 1, 1, w)
        H = torch.mean(concatenated, dim=[1, 3])[0]
        H = self.Hdense1(H)
        H= torch.relu(H)
        H = torch.sigmoid(self.Hdense2(H))
        H = H.view(1, 1, h, 1)
        concatenated = concatenated * (C + H + W)
        concatenated = torch.mean(concatenated, dim=3,keepdim=True)
        concatenated = torch.cat([concatenated, w0], dim=1)
        concatenated = self.convs1x1two(concatenated)
        concatenated = torch.cat([cash_bias, concatenated], dim=2)
        concatenated = torch.relu(concatenated)
        concatenated = concatenated.view(ifstates,14)
        concatenated = torch.softmax(concatenated, dim=1)
        return concatenated
```
the init function takes parameters such as number of assets and defines the layers of the network.

the forward function takes the observation which is a 3D matrix then runs it through the following network:
![self.data](readme_images\cnnarch.png)
>this image shows the architecture of the network used from input to output.

### c) creating the agent
the agent is a class that will be used to train the policy network.
```python
def compute_returns(rewards, gamma):
    returns = 0
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
    returns=R
    return returns
def train(env, policy, optimizer, gamma, num_episodes):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latest_checkpoint = max(glob.glob(f"checkpoints/checkpoint_*.pt"), key=os.path.getctime, default=None)
    if latest_checkpoint is not None:
        print(f"Loading checkpoint: {latest_checkpoint}")
        policy.load_state_dict(torch.load(latest_checkpoint))
    
    for i in range(num_episodes):
        state = env.reset()
        done = False
        rewards = []
        
        while not done:
            action = policy(torch.tensor(state).permute(2,0,1).unsqueeze(0).float())
            state, reward, done, _ = env.step(action.cpu().detach().numpy()[0])
            rewards.append(reward)
            
        returns = torch.tensor(compute_returns(rewards, gamma), requires_grad=True)
        loss = -returns
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #scheduler.step()  # Update the learning rate

        
        if i % 10 == 0:
            print(f"Episode {i}, loss: {loss.item()}")
        if i % 200 == 0:
            # Save the model parameters to a file
            checkpoint_path = f"checkpoints/checkpoint_{i}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
        # Define the checkpoint directory

#scheduler = lr_scheduler.StepLR(torch.optim.Adam(policynet.parameters(), lr=0.01), step_size=1000, gamma=0.1)
train(env, policynet, torch.optim.Adam(policynet.parameters(), lr=0.01),1, 100)
```
this is a vanilla policy gradient agent. it takes the environment, the policy network, the optimizer, the discount factor and the number of episodes. it generates an action using the policy network, then takes the reward and the next observation from the environment. it then calculates the return and the loss and updates the policy network.
