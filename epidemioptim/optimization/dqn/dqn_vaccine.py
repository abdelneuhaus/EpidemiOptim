"""
Adapted from https://github.com/higgsfield/RL-Adventure
"""


import os

import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.autograd as ag

from epidemioptim.optimization.shared.replay_buffer import ReplayBuffer
from epidemioptim.optimization.shared.rollout_vaccine import run_rollout
from epidemioptim.optimization.base_algorithm import BaseAlgorithm
from epidemioptim.optimization.shared.networks import Critic
from epidemioptim.utils import Logger

try:
    import sobol_seq
    def sample_goals(n, dim_goal):
        return sobol_seq.i4_sobol_generate(dim_goal, n)
except:
    def sample_goals(n, dim_goal):
        return np.random.uniform(0, 1, size=(n, dim_goal))


class DQN_vaccine(BaseAlgorithm):
    def __init__(self, env, params):
        """
        DQN algorithm from Mnih et al., 2015.
        This implementation includes mechanisms from Universal Value Function Approximators (Schaul et al., 2015)
        and Agent57 (Badia et al., 2019).

        Parameters
        ----------
        env: BaseEnv
            Learning environment.
        params: dict
            Dictionary of parameters.

        Attributes
        ----------
        batch_size: int
            Batch size.
        gamma: float
            Discount factor in [0, 1].
        layers: tuple of ints
            Describes sizes of hidden layers of the critics.
        goal_conditioned: bool
            Whether the algorithm is goal conditioned or not. This is the idea behind UVFA (Schaul et al., 2015).
        replace_target_cnt: int
            Frequency with which target critics should be replaced by current critics (in learning steps).
        logdir: str
            Logging directory
        save_policy_every: int
            Frequency to save policy (in episodes).
        eval_and_log_every: int
            Frequency to pring logs (in episodes).
        n_evals_if_stochastic: int
            Number of evaluation episodes if the environment is stochastic.
        stochastic: bool
            Whether the environment is stochastic.
        epsilon: float
            Probability to sample a random action (epsilon-greedy exploration). This is set to 0 during evaluation.
        dims: dict
            Dimensions of states and actions.
        cost_function: BaseMultiCostFunction
            Multi-cost function.
        nb_costs: int
            Number of cost functions
        use_constraints: bool
            Whether the algorithm uses constraints.
        """
        super(DQN_vaccine, self).__init__(env, params)

        # Save parameters
        self.batch_size = self.algo_params['batch_size']
        self.gamma = self.algo_params['gamma']
        self.layers = tuple(self.algo_params['layers'])
        self.goal_conditioned = self.algo_params['goal_conditioned']
        self.replace_target_cnt = self.algo_params['replace_target_count']
        self.logdir = params['logdir']
        self.save_policy_every = self.algo_params['save_policy_every']
        self.eval_and_log_every = self.algo_params['eval_and_log_every']
        self.n_evals_if_stochastic = self.algo_params['n_evals_if_stochastic']
        self.stochastic = params['model_params']['stochastic']
        self.epsilon = self.algo_params['epsilon_greedy']
        self.cost_function = self.env.unwrapped.cost_function
        self.nb_costs = 1#self.env.unwrapped.cost_function.nb_costs
        self.use_constraints = self.cost_function.use_constraints
        self.is_multi_obj = True if self.goal_conditioned else False  # DQN is not a multi-obj algorithm, unless it is goal-conditioned
        self.dims = dict(s=env.observation_space.shape[0],
                         a=env.action_space.n)


        if self.goal_conditioned:
            self.goal_dim = self.env.unwrapped.cost_function.goal_dim
            eval_goals = self.cost_function.get_eval_goals(1)
            goals, index, inverse = np.unique(eval_goals, return_inverse=True, return_index=True, axis=0)
            goal_keys = [str(g) for g in goals]
        else:
            self.goal_dim = 0
            goal_keys = [str(self.cost_function.action)]

        # Initialize Logger.
        if self.logdir:
            os.makedirs(self.logdir + 'models/', exist_ok=True)
            stats_keys = ['mean_agg', 'std_agg'] + ['mean_C{}'.format(i) for i in range(self.nb_costs)] + ['std_C{}'.format(i) for i in range(self.nb_costs)]

            keys = ['Episode', 'Best score so far', 'Eval score']
            for k in goal_keys:
                for s in stats_keys:
                    keys.append('Eval, g: ' + k + ': ' + s)
            keys += ['Loss {}'.format(i + 1) for i in range(self.nb_costs)] + ['Train, Cost {}'.format(i + 1) for i in range(self.nb_costs)] + ['Train, Aggregated cost']
            self.logger = Logger(keys=keys,
                                 logdir=self.logdir)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.algo_params['buffer_size'])

        # Initialize critics
        self.Q_eval = Critic(n_critics=self.nb_costs,
                             dim_state=self.dims['s'],
                             dim_goal=self.goal_dim,
                             dim_actions=self.dims['a'],
                             goal_ids=((), ()),  # no goal, the mixing parameters comes after the critic
                             layers=self.layers)
        self.Q_next = Critic(n_critics=self.nb_costs,
                             dim_state=self.dims['s'],
                             dim_goal=self.goal_dim,
                             dim_actions=self.dims['a'],
                             goal_ids=((), ()),
                             layers=self.layers)

        # Initialize optimizers.
        self.optimizers = [optim.Adam(q.parameters(), lr=self.algo_params['lr']) for q in self.Q_eval.qs]

        # If we use constraint, we train a Q-network per constraint using a negative reward of -1 whenever the constraint is violated.
        # This network learns to estimate the number of times the constraint will be violated in the future.
        # We then use it to guide action selection, selecting the action that does not violate the constraint when the other does,
        # or the action that minimizes constraint violation when both action lead to constraint violations.
        self.nb_constraints = 0

        # Initialize counters
        self.learn_step_counter = 0
        self.env_step_counter = 0
        self.episode = 0
        self.best_cost = np.inf
        self.aggregated_costs = []
        self.costs = []

    def _replace_target_network(self):
        """
        Replaces the target network with the evaluation network every 'self.replace_target_cnt' learning steps.
        """
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.set_goal_params(self.Q_eval.get_params())
        

    def _update(self, batch_size):
        """
        Performs network updates according to the DQN algorithm.
        Here we update several critics: one for each cost and one for each constraint.
        We then use these critics at decision time: first filtering actions that are not expected to violate constraints,
        then selecting actions that maximize a convex combination of the costs as expressed by the mixing parameter beta.
        Beta can be provided by the experimenter (dqn) or selected by the agent (goal_dqn).

        Parameters
        ----------
        batch_size: int
            Batch size

        Returns
        -------
        loss

        """

        # Reset gradients of optimizes
        for opt in self.optimizers:
            opt.zero_grad()

        # Update target network.
        self._replace_target_network()

        # Sample a batch
        state, action, cost_aggregated, costs, next_state, goal, done, constraints = self.replay_buffer.sample(batch_size)

        # Concatenate goal if the policy is goal conditioned (might not be used afterwards).
        state = ag.Variable(torch.FloatTensor(np.float32(state)))
        #print("state", state)
        next_state = ag.Variable(torch.FloatTensor(np.float32(next_state)))
        action = ag.Variable(torch.LongTensor(action))
        indices = np.arange(self.batch_size)
        rewards = [- ag.Variable(torch.FloatTensor(c_func.scale(c))) for c_func, c in zip(self.cost_function.costs, costs.transpose())]


        q_preds = self.Q_eval.forward(state)
        q_preds = [q_p[indices, action] for q_p in q_preds]
        q_nexts = self.Q_next.forward(next_state)
        q_evals = self.Q_eval.forward(next_state)

        max_actions = [torch.argmax(q_ev, dim=1) for q_ev in q_evals]
        #print(q_preds, q_nexts, q_evals)
        q_targets = [r + self.gamma * q_nex[indices, max_act] for r, q_nex, max_act in zip(rewards, q_nexts, max_actions)]
        losses = [(q_pre - ag.Variable(q_targ.data)).pow(2).mean() for q_pre, q_targ in zip(q_preds, q_targets)]
        for loss in losses:
            loss.backward()

        for opt in self.optimizers:
            opt.step()

        self.learn_step_counter += 1
        return losses

    def store_episodes(self, episodes):
        lengths = []
        for e in episodes:
            for t in range(e['env_states'].shape[0] - 1):
                self.replay_buffer.push(state=e['env_states'][t],
                                        action=e['actions'][t],
                                        aggregated_cost=e['aggregated_costs'][t][0],
                                        costs=e['costs'],
                                        next_state=e['env_states'][t + 1],
                                        constraints=None,
                                        goal=e['goal'],
                                        done=e['dones'][t])
            lengths.append(e['env_states'].shape[0] - 1)
        return lengths

    def act(self, state, deterministic=False):
        """
        Policy that uses the learned critics.

        Parameters
        ----------
        state: 1D nd.array
            Current state.
        deterministic: bool
            Whether the policy should be deterministic (e.g. in evaluation mode).

        Returns
        -------
        action: nd.array
            Action vector.
        q_constraints: nd.array
            Values of the critics estimating the expected constraint violations.
        """
        state=state.tolist()
        state=torch.FloatTensor(state)
        if np.random.rand() > self.epsilon or deterministic:
            with torch.no_grad():
                state = ag.Variable(torch.FloatTensor(state).unsqueeze(0))
                q_value = self.Q_eval.forward(state)[0]
            action = int(q_value.max(1)[1].data[0])
            q_constraints = None
        else:
            # Epsilon-greedy exploration, random action with probability epsilon.
            action = np.random.randint(self.dims['a'])
            q_constraints = None
        return np.atleast_1d(action), q_constraints

    def update(self):
        """
        Update the algorithm.

        """
        if self.env_step_counter > 0:
            losses = self._update(self.batch_size)
            return [np.atleast_1d(l.data)[0] for l in losses]
        else:
            return [0] * (2 + self.nb_constraints)

    def save_model(self, path):
        """
        Extract model state dicts and save them.
        Parameters
        ----------
        path: str
            Saving path.

        """
        q_eval = self.Q_eval.get_model()
        to_save = [q_eval]
        if self.use_constraints:
            q_constraints = self.Q_eval_constraints.get_model()
            to_save.append(q_constraints)
        with open(path, 'wb') as f:
            torch.save(to_save, f)

    def load_model(self, path):
        """
        Load model from file and feed critics' state dicts.
        Parameters
        ----------
        path: str
            Loading path
        """
        with open(path, 'rb') as f:
            out = torch.load(f)
        try:
            self.Q_eval.set_model(out[0])
        except:
            self.Q_eval.set_model(out)

        if self.use_constraints:
            self.Q_eval_constraints.set_model(out[1])

    def learn(self, num_train_steps):
        """
        Main training loop.

        Parameters
        ----------
        num_train_steps: int
            Number of training steps (environment steps)

        Returns
        -------

        """

        while self.env_step_counter < num_train_steps:
            if self.goal_conditioned:
                goal = self.env.unwrapped.sample_cost_function_params()
            else:
                goal = None
            episodes = run_rollout(policy=self,
                                   env=self.env,
                                   n=1,
                                   goal=goal,
                                   eval=False,
                                   additional_keys=('costs'),
                                   )
            lengths = self.store_episodes(episodes)
            self.env_step_counter += np.sum(lengths)
            self.episode += 1

            self.aggregated_costs.append(np.sum(episodes[0]['aggregated_costs']))
            self.costs.append(np.sum(episodes[0]['costs'], axis=0))

            # Update
            if len(self.replay_buffer) > self.batch_size:
                update_losses = []
                for _ in range(int(np.sum(lengths) * 0.5)):
                    update_losses.append(self.update())
                update_losses = np.array(update_losses)
                losses = update_losses.mean(axis=0)
            else:
                losses = [np.nan] * 2

            if self.episode % self.eval_and_log_every == 0:
                # Run evaluations
                new_logs, eval_costs = self.evaluate(n=self.n_evals_if_stochastic if self.stochastic else 1)
                # Compute train scores
                train_agg_cost = np.mean(self.aggregated_costs)
                train_costs = np.array(self.costs).mean(axis=0)
                self.log(self.episode, new_logs, losses, train_agg_cost, train_costs)
                # Reset training score tracking
                self.aggregated_costs = []
                self.costs = []

            if self.episode % self.save_policy_every == 0:
                self.save_model(self.logdir + '/models/policy_{}.cp'.format(self.episode))
        print('Run has terminated successfully')

    def evaluate(self, n=None, goal=None, best=None, reset_same_model=False):
        # run eval
        if n is None:
            n = self.n_evals_if_stochastic if self.env.unwrapped.stochastic else 1
        if self.goal_conditioned:
            if goal is not None:
                eval_goals = np.array([goal] * n)
            else:
                eval_goals = self.cost_function.get_eval_goals(n)

            n = eval_goals.shape[0]
        else:
            eval_goals = None
        eval_episodes = run_rollout(policy=self,
                                    env=self.env,
                                    n=n,
                                    goal=eval_goals,
                                    eval=True,
                                    reset_same_model=reset_same_model,
                                    additional_keys=('costs'),
                                    )
        new_logs, costs = self.compute_eval_score(eval_episodes, eval_goals)
        return new_logs, costs

    def compute_eval_score(self, eval_episodes, eval_goals):
        aggregated_costs = [np.sum(e['aggregated_costs']) for e in eval_episodes]
        costs = np.array([np.sum(e['costs'], axis=0) for e in eval_episodes])
        new_logs = dict()
        costs_mean = np.mean(np.atleast_2d(costs), axis=0)
        costs_std = np.std(np.atleast_2d(costs), axis=0)
        for i_r in range(self.nb_costs):
            new_logs['Eval, g: ' + str([]) + ': ' + 'mean_C{}'.format(i_r)] = costs_mean[i_r]
            new_logs['Eval, g: ' + str([]) + ': ' + 'std_C{}'.format(i_r)] = costs_std[i_r]
        new_logs['Eval score'] = np.mean(costs)
        new_logs['Eval, g: ' + str([]) + ': ' + 'mean_agg'] = np.mean(aggregated_costs)
        new_logs['Eval, g: ' + str([]) + ': ' + 'std_agg'] = np.mean(aggregated_costs)
        return new_logs, costs

    def log(self, episode, new_logs, losses, train_agg_cost, train_costs):
        if new_logs['Eval score'] < self.best_cost:
            self.best_cost = new_logs['Eval score']
            self.save_model(self.logdir + '/models/best_model.cp')

        train_log_dict = {'Episode': episode, 'Best score so far': self.best_cost}

        for i in range(self.nb_costs):
            train_log_dict['Loss {}'.format(i + 1)] = losses[i]
            train_log_dict['Train, Cost {}'.format(i + 1)] = train_costs
            train_log_dict['Train, Aggregated cost'] = train_agg_cost

        new_logs.update(train_log_dict)
        self.logger.add(new_logs)
        self.logger.print_last()
        self.logger.save()
