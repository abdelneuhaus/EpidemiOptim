import numpy as np
import gym
import math
from epidemioptim.environments.gym_envs.base_env import BaseEnv
from epidemioptim.utils import *
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class EpidemicVaccination(BaseEnv):
    def __init__(self,
                 cost_function,
                 model,
                 simulation_horizon,
                 ratio_death_to_R=0.0001,  # death ratio among people who were infected
                 time_resolution=30,
                 seed=np.random.randint(1e6)
                 ):
        """
        EpidemicDiscrete environment is based on the Epidemiological SEIRAH model from Prague et al., 2020 and on a bi-objective
        cost function (death toll and gdp recess).

        Parameters
        ----------
        cost_function: BaseCostFunction
            A cost function.
        model: BaseModel
            An epidemiological model.
        simulation_horizon: int
            Simulation horizon in days.
        ratio_death_to_R: float
            Ratio of deaths among recovered individuals.
        time_resolution: int
            In days.
        """

        # Initialize model
        self.model = model
        self.stochastic = self.model.stochastic
        self.simulation_horizon = simulation_horizon
        self.reset_same = False  # whether the next reset resets the same epidemiological model

        # Initialize cost function
        self.cost_function = cost_function
        self.nb_costs = 1
        self.cumulative_costs = [0 for _ in range(self.nb_costs)]

        # Initialize states
        self.state_labels = self.model._age_groups + ['previous_politic', 'current_vaccination_politic'] + \
            ['cumulative_cost_{}'.format(id_cost) for id_cost in range(self.nb_costs)]
        self.label_to_id = dict(zip(self.state_labels, np.arange(len(self.state_labels))))
        # To modify
        self.normalization_factors = [4191481.438] * len(self.model._age_groups) + \
                                     [[0,0,0], [0,0,0], 150, 1]


        super().__init__(cost_function=cost_function,
                         model=model,
                         simulation_horizon=simulation_horizon,
                         dim_action=8,
                         discrete=True,
                         seed=seed)

        self.ratio_death_to_R = ratio_death_to_R
        self.time_resolution = time_resolution
        self._max_episode_steps = simulation_horizon // time_resolution
        self.history = None

        # Vaccination parameters
        self.sigma = []
        self.vaccine_groups = self.get_pop_vaccine_groups()


    def get_pop_vaccine_groups(self):
        """
        Return the total number of people in the 3 neo-groups (S compartment only)
        """
        a = ['0-4', '5-9', '10-14', '15-19']
        b = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54']
        c = ['55-59', '60-64', '65-69',  '70-74', '75+']
        groups = ['0-19', '20-54', '55+']
        vaccine_groups = {}
        for i in groups :
            vaccine_groups[i] = {}
        sumA, sumB, sumC = 0, 0, 0
        for i in a:
            sumA += (self.model.current_state[i]['S1'] + self.model.current_state[i]['S2'] + self.model.current_state[i]['S3'] + self.model.current_state[i]['S4'])
        for i in b:
            sumB += (self.model.current_state[i]['S1'] + self.model.current_state[i]['S2'] + self.model.current_state[i]['S3'] + self.model.current_state[i]['S4'])
        for i in c:
            sumC += (self.model.current_state[i]['S1'] + self.model.current_state[i]['S2'] + self.model.current_state[i]['S3'] + self.model.current_state[i]['S4'])
        vaccine_groups['0-19']['S'] = sumA
        vaccine_groups['20-54']['S'] = sumB
        vaccine_groups['55+']['S'] = sumC
        return vaccine_groups


    def _compute_delta_coverage(self):
        maxcoverage = [x*100 for x in self.model._vaccination_coverage]
        _deltaCoverage = list(range(len(maxcoverage)))
        _deltaCoverage[0] = maxcoverage[0]
        for i in range(1, len(maxcoverage)):
            if maxcoverage[i] != maxcoverage[i-1]:
                _deltaCoverage[i] = maxcoverage[i] - maxcoverage[i-1]
            else:
                _deltaCoverage[i] = maxcoverage[i-1]
        for i in range(len(_deltaCoverage)):
            if _deltaCoverage[i] == 0:
                _deltaCoverage[i] = 10e-6 
        _deltaCoverage[-1] = _deltaCoverage[-2]
        return _deltaCoverage
    

    def initialize_model_for_vaccine(self):
        simulation_horizon = 370
        model_states = []
        for i in range(simulation_horizon):
            model_state = self.model.run_n_steps()
            model_states += model_state.tolist()
            self.model_state = self.model._get_current_state()
        return self.model.current_state, self.model.current_internal_params
        

    def compute_sigma(self):
        mwl = self.mitigation_windows[self.step]
        lowVP = 1
        pi = lowVP*(self.vaccination_coverage[self.vacStep]/100)
        classes = ['S1', 'S2', 'S3', 'S4']
        popGrp = ['S1', 'S2', 'S3', 'S4', 'E21', 'E22', 'E23', 'E31', 'E32', 'E33', 'E41', 'E42', 
                  'E43', 'V11', 'V21', 'V31', 'V41', 'V12', 'V22', 'V32', 'V42', 'I2', 'I3', 'I4']
        sigma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        wcv, Ntot = 0, 0
        for n in range(16):
            Ntot += sum([self.current_state[self._age_groups[n]]['{}'.format(s)] for s in popGrp])
            if self.whovaccinated[self.vacStep+1][n] == 1:
                wcv += sum([self.current_state[self._age_groups[n]]['{}'.format(s)] for s in classes])
        g = (pi*Ntot/wcv)
        if g>1:
            g=1
        for k in range(16):
            if self.whovaccinated[self.vacStep+1][k] == 1:
                sigma[k] = 1/mwl*(-math.log(1-g))
            if sigma[k] < 0:
                sigma[k] = 0
        for f in range(16):
            size = sum([self.current_state[self._age_groups[f]]['{}'.format(s)] for s in popGrp])
            if self.dCV1[f]/size >= self.coverage_threshold[f]/0.8944:
                sigma[f] = 0
        return sigma


    def _update_previous_env_state(self):
        """
        Save previous env state.

        """
        if self.env_state is not None:
            self.previous_env_state = self.env_state.copy()
            self.previous_env_state_labelled = self.env_state_labelled.copy()


    def _update_env_state(self):
        """
        Update the environment state.

        """

        # Update env state
        self.env_state_labelled = dict()
        for i in self.model._age_groups:
            self.env_state_labelled[i] = dict(zip(self.model.internal_states_labels, [self.model.initial_state[i]['{}0'.format(s)] for s in self.model.internal_states_labels]))
            
        #self.env_state_labelled = dict(zip(self.model.internal_states_labels, self.model_state))
        self.env_state_labelled.update(previous_politic=self.previous_politic,
                                       current_vaccination_politic=self.vaccination_politic
                                       )
        # track cumulative costs in the state.
        for id_cost in range(self.nb_costs):
            self.env_state_labelled['cumulative_cost_{}'.format(id_cost)] = self.cumulative_costs[id_cost]
        assert sorted(list(self.env_state_labelled.keys())) == sorted(self.state_labels), "labels do not match"
        self.env_state = np.array([self.env_state_labelled[k] for k in self.state_labels])

        # Set previous env state to env state if first step
        if self.previous_env_state is None:
            # happens at first step
            self.previous_env_state = self.env_state.copy()
            self.previous_env_state_labelled = self.env_state_labelled.copy()

    def reset_same_model(self):
        """
        To call if you want to reset to the same model the next time you call reset.
        Will be cancelled after the first reset, it needs to be called again each time.

        """
        self.reset_same = True

    def reset(self):
        """
        Reset the environment and the tracking of data.

        Returns
        -------
        nd.array
            The initial environment state.

        """
        # initialize history of states, internal model states, actions, cost_functions, deaths
        self.history = dict(env_states=[],
                            model_states=[],
                            env_timesteps=[],
                            actions=[],
                            aggregated_costs=[],
                            costs=[],
                            vaccinations=[],
                            deaths=[])
        # initialize time and lockdown days counter
        self.t = 0
        self.count_deaths = 0
        self.count_since_start_politic = 0
        self.count_since_last_politic = 0

        self.vaccination_politic = [0,0,1]
        self.previous_politic = self.vaccination_politic
        self.cumulative_costs = [0 for _ in range(self.nb_costs)]

        # initialize model internal state and params
        if self.reset_same:
            self.model.reset_same_model()
            self.reset_same = False
        else:
            self.model.reset()
        self.model_state = self.model._get_current_state()

        self._update_previous_env_state()
        self._update_env_state()

        self.history['env_states'].append(self.env_state.copy())
        self.history['model_states'].append(self.model_state.copy())
        self.history['env_timesteps'].append(self.t)

        return self._normalize_env_state(self.env_state)


    def update_with_action(self, action):
        """
        Implement effect of vaccination on vaccination rate.

        Parameters
        ----------
        action: int
            Action is a list with 0 (no vaccination) or 1 (vaccination).

        """

        # Translate actions
        self.previous_politic = self.vaccination_politic
        
        for i in range(len(action)):
            if action[i] == 0:
                # no vaccination
                self.jump_of = min(self.time_resolution, self.simulation_horizon - self.t)
                self.vaccination_politic[i] = 0
                if self.previous_politic[i] == self.vaccination_politic[i]:
                    self.count_since_last_politic += self.jump_of
                else:
                    self.count_since_last_politic = self.jump_of
                    self.count_since_start_politic = 0
            else:
                self.jump_of = min(self.time_resolution, self.simulation_horizon - self.t)
                self.vaccination_politic[i] = 1
                if self.vaccination_politic[i] == self.previous_politic[i]:
                    self.count_since_start_politic += self.jump_of
                else:
                    self.count_since_start_politic = self.jump_of
                    self.count_since_last_politic = 0

        # Modify model parameters based on lockdown state
        self.sigma = self._compute_sigma()
        for i in range(16):
            if i in [0,1,2,3]:
                self.model.current_internal_params['sigma'][i] = self.sigma[0][0]
            elif i in [4,5,6,7,8,9,10]:
                self.model.current_internal_params['sigma'][i] = self.sigma[0][1]
            else:
                self.model.current_internal_params['sigma'][i] = self.sigma[0][2]


    def step(self, action):
        """
        Traditional step function from OpenAI Gym envs. Uses the action to update the environment.

        Parameters
        ----------
        action: int
            Action is a list with 0 (no vaccination) or 1 (vaccination).

        Returns
        -------
        state: nd.array
            New environment state.
        cost_aggregated: float
            Aggregated measure of the cost.
        done: bool
            Whether the episode is terminated.
        info: dict
            Further infos. In our case, the costs, icu capacity of the region and whether constraints are violated.

        """
        action = list(action)
        self.update_with_action(action)
        
        # Run model for jump_of steps
        model_state = [self.model_state]
        model_states = []
        for sigma in self.sigma:
            model_state = self.model.run_n_steps(model_state[-1], 1)
            model_states += model_state.tolist()
        self.model_state = model_state[-1]  # last internal state is the new current one
        self.t += self.jump_of

        # Update state
        self._update_previous_env_state()
        self._update_env_state()

        # Store history
        costs = self.cost_function.compute_cost(previous_state=np.atleast_2d(self.previous_env_state),
                                                state=np.atleast_2d(self.env_state),
                                                label_to_id=self.label_to_id,
                                                action=action,
                                                others=dict(jump_of=self.time_resolution))

        self.cumulative_costs[0] += costs
        n_deaths = self.cost_function.compute_cost(previous_state=np.atleast_2d(self.previous_env_state),
                                                     state=np.atleast_2d(self.env_state),
                                                     label_to_id=self.label_to_id,
                                                     action=action)

        self._update_env_state()

        self.history['actions'] += [action] * self.jump_of
        self.history['env_states'] += [self.env_state.copy()] * self.jump_of
        self.history['env_timesteps'] += list(range(self.t - self.jump_of, self.t))
        self.history['model_states'] += model_states
        self.history['vaccinations'] += [self.vaccination_politic] * self.jump_of
        self.history['deaths'] += [n_deaths / self.jump_of] * self.jump_of

        # Compute cost_function
        self.history['costs'] += [costs / self.jump_of for _ in range(self.jump_of)]
        self.costs = costs.copy()

        if self.t >= self.simulation_horizon:
            done = 1
        else:
            done = 0

        return self._normalize_env_state(self.env_state), costs, done

    # Utils
    def _normalize_env_state(self, env_state):
        return env_state.copy()

    def _set_rew_params(self, goal):
        self.cost_function.set_goal_params(goal.copy())

    def sample_cost_function_params(self):
        return self.cost_function.sample_goal_params()

    # Format data for plotting
    def get_data(self):
        data = dict(history=self.history.copy(),
                    time_jump=1,
                    model_states_labels=self.model.internal_states_labels
                    )
        t = self.history['env_timesteps']
        cumulative_death = [np.sum(self.history['deaths'][:i]) for i in range(len(t) - 1)]
        costs = np.array(self.history['costs'])
        to_plot = [np.array(self.history['deaths']),
                   np.array(cumulative_death),
                   costs[0],
                   np.array(self.history['vaccinations'])
                   ]
        labels = ['New Deaths', 'Total Deaths', r'Aggregated Cost', 'Transmission rate']
        legends = [None, None, None, None]
        stats_run = dict(to_plot=to_plot,
                         labels=labels,
                         legends=legends)
        data['stats_run'] = stats_run
        data['title'] = 'Death Cost: {}, Aggregated Cost: {:.2f}'.format(int(cumulative_death[-1]),
                                                                                             np.sum(self.history['aggregated_costs']))
        return data


if __name__ == '__main__':
    from epidemioptim.utils import plot_stats, plot_preds
    from epidemioptim.environments.cost_functions import get_cost_function
    from epidemioptim.environments.models import get_model

    simulation_horizon = 361
    model = get_model(model_id='heffernan_model', params=dict(stochastic=False))
    cost_func = get_cost_function(cost_function_id='death_toll_cost_vaccine', params=dict(id_cost=0, ratio_death_to_R=0.0001))
    env = gym.make('EpidemicVaccination-v0',
                    cost_function=cost_func,
                    model=model,
                    simulation_horizon=simulation_horizon)
    env.reset()
    model.current_state, model.current_internal_params = env.initialize_model_for_vaccine()
    model_states = []
    for i in range(simulation_horizon):
        model_state = model.run_n_steps()
        model_states += model_state.tolist()
    
    # Plot
    time = np.arange(simulation_horizon)
    # plot_preds(t=time,
    #            states=np.array(model_states).transpose()[13]+np.array(model_states).transpose()[14]+np.array(model_states).transpose()[15]+np.array(model_states).transpose()[16])
    plot_preds(t=time,states=np.array(model_states).transpose()[23], title="Vaccination")


    # # To delete if no GDP recess cost function
    # N_region = model.pop_size[age_group]
    # N_country = np.sum(list(model.pop_size.values()))
    # ratio_death_to_R = 0.0001

    # #Actions
    # actions = ([0,0,1], [0,0,1], [0,1,1], [0,1,1], [0,1,1], [0,1,1])
    # t = 0
    # r = 0
    # done = False
    # while not done:
    #     out = env.step(actions[1])
    #     t += 1
    #     r += out[1]
    #     done = out[2]
    # stats = env.unwrapped.get_data()

    # plot model states
    # plot_stats(t=stats['history']['env_timesteps'],
    #            states=np.array(stats['history']['model_states']).transpose(),
    #            labels=stats['model_states_labels'],
    #            vaccination=np.array(stats['history']['vaccinations']),
    #            time_jump=stats['time_jump'])
    # plot_stats(t=stats['history']['env_timesteps'][1:],
    #            states=stats['stats_run']['to_plot'],
    #            labels=stats['stats_run']['labels'],
    #            legends=stats['stats_run']['legends'],
    #            title=stats['title'],
    #            vaccination=np.array(stats['history']['vaccinations']),
    #            time_jump=stats['time_jump'],
    #            show=True
    #            )
