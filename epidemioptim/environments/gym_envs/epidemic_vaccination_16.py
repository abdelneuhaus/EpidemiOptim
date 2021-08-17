import numpy as np
import gym
import math
import itertools
from epidemioptim.environments.gym_envs.base_env import BaseEnv
from epidemioptim.utils import *
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class EpidemicVaccinationMultiGroups(BaseEnv):
    def __init__(self,
                 cost_function,
                 model,
                 simulation_horizon,
                 ratio_death_to_R=0.0016,  # death ratio among people who were infected, don't used
                 time_resolution=15,
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
        self.factors = np.repeat(np.array(self.model._pop_size.tolist()), 32)
        self.normalization_factors = [self.factors.tolist()+ [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,67063703]][0]


        super().__init__(cost_function=cost_function,
                         model=model,
                         simulation_horizon=simulation_horizon,
                         dim_action=65536,
                         discrete=True,
                         seed=seed)

        self.ratio_death_to_R = ratio_death_to_R
        self.time_resolution = time_resolution
        self._max_episode_steps = simulation_horizon // time_resolution
        self.history = None

        # Vaccination parameters
        self.sigma = []
        self.vacStep = 0
        self.vaccine_groups = self.get_S_pop_vaccine_groups()
        self.vaccination_current_politic = self.who_can_vaccinate_3_groups()
        self.coverage_goals = [1.36, 2.50, 9.90, 16.40, 27.79, 34.30, 45.68, 52.19, 63.58, 63.58, 63.58]
        self._transpose = list(itertools.product([0, 1], repeat=16))
        self.transpose_actions = tuple(np.array(self._transpose).tolist())



    def get_S_pop_vaccine_groups(self):
        """
        Return the total number of people in the 3 neo-groups (S compartment only)
        """
        a = ['0-4', '5-9', '10-14', '15-19','20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64','65-69',  '70-74', '75+']
        classes = ['S1', 'S2', 'S3', 'S4']
        vaccine_groups = []
        for i in a:
            sumA = 0
            sumA = sum([self.model.current_state[i]['{}'.format(s)] for s in classes])
            vaccine_groups.append(sumA)
        return vaccine_groups


    def get_total_pop_vaccine_groups(self):
        """
        Return the total number of people in the 3 neo-groups (S compartment only)
        """
        a = ['0-4', '5-9', '10-14', '15-19','20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64','65-69',  '70-74', '75+']
        classes = ['S1', 'S2', 'S3', 'S4', 'E21', 'E22', 'E23', 'E31', 'E32', 'E33', 'E41', 'E42', 
                   'E43', 'V11', 'V21', 'V31', 'V41', 'V12', 'V22', 'V32', 'V42', 'I2', 'I3', 'I4']
        total_pop = []
        for i in a:
            sumA = 0
            sumA = sum([self.model.current_state[i]['{}'.format(s)] for s in classes])
            total_pop.append(sumA)
        return total_pop

    def get_dcv1_and_threshold(self):
        """
        Return the total number of people in the 3 neo-groups (S compartment only)
        """
        threshold_groups = self.model.coverage_threshold  
        dcv1_groups = []
        for i in range(16):
            dcv1_groups.append(self.model.dCV1[i])
        return dcv1_groups, threshold_groups


    def _compute_delta_coverage(self):
        """
        Compute the vaccination coverage for each time period regarding Excel file
        """
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
        """
        Simulate the epidemic until 5th January, 2021 (beginning of the French vaccination campaign)
        """
        simulation_horizon = 371
        model_states = []
        for i in range(simulation_horizon):
            model_state = self.model.run_n_steps()
            model_states += model_state.tolist()
            self.model_state = self.model._get_current_state()
        return self.model.current_state, self.model.current_internal_params , model_states
    

    def who_can_vaccinate_3_groups(self):
        """
        Compute each 15 days if a group is eligible to the vaccination
        """
        vaccination_program = []
        data = self.model.whovaccinated
        for i in data:
            for j in i:
                tmp = []
                if 1 in i[0:16]:
                    tmp.append(1)
                else:
                    tmp.append(0)
            vaccination_program.append(tmp)
        return np.array(vaccination_program)


    def compute_sigma_with_action(self, action):
        """
        Compute the vaccination rate (same as in the model)
        """
        mwl = self.model.mitigation_windows[self.model.step-1]
        lowVP = 1
        pi = lowVP*(self.model.vaccination_coverage[self.model.vacStep]/100)
        popGrp = ['S1', 'S2', 'S3', 'S4', 'E21', 'E22', 'E23', 'E31', 'E32', 'E33', 'E41', 'E42', 
                  'E43', 'V11', 'V21', 'V31', 'V41', 'V12', 'V22', 'V32', 'V42', 'I2', 'I3', 'I4']
        sigma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        wcv, Ntot = 0, 0
        for n in range(16):
            Ntot += sum([self.model.current_state[self.model._age_groups[n]]['{}'.format(s)] for s in popGrp])
        for n in range(16):
            if action[n] == 1:
                wcv += self.get_S_pop_vaccine_groups()[n]
        g = (pi*Ntot/wcv)
        if g>1:
            g=0.99999999
        sig = 1/mwl*(-math.log(1-g))
        for k in range(16):
            if action[k] == 1:
                sigma[k] = sig
            if sigma[k] < 0:
                sigma[k] = 0
        total_pop = self.get_total_pop_vaccine_groups()
        dcv1, threshold = self.get_dcv1_and_threshold()
        for f in range(16):
            size = total_pop[f]
            if dcv1[f]/size >= threshold[f]/0.8944:
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
            self.env_state_labelled[i] = dict(zip(self.model.internal_states_labels, [self.model.current_state[i]['{}'.format(s)] for s in self.model.internal_states_labels]))
        self.env_state_labelled.update(previous_politic=self.previous_politic,
                                       current_vaccination_politic=self.vaccination_politic
                                       )
        # track cumulative costs in the state.
        for id_cost in range(self.nb_costs):
            self.env_state_labelled['cumulative_cost_{}'.format(id_cost)] = self.cumulative_costs[id_cost]
        assert sorted(list(self.env_state_labelled.keys())) == sorted(self.state_labels), "labels do not match"

        self.env_state = np.array([self.env_state_labelled[k] for k in self.state_labels])
        for i in range(16):
            self.env_state[i] = [self.env_state[i][j] for j in self.model.internal_states_labels]
        merge = []
        for i in self.env_state:
            if type(i) != list():
                i = as_list(i)
            merge = merge + i
        self.env_state = np.array(merge)
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

        self.vaccination_politic = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]
        self.previous_politic = self.vaccination_politic
        self.cumulative_costs = [0 for _ in range(self.nb_costs)]

        # initialize model internal state and params
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
        for i in range(16):
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

        # Modify model parameters based on state
        sigma = self.compute_sigma_with_action(action)
        self.model.current_internal_params['sigma'] = np.array(sigma)
        if self.model.t > 670:
            for j in range(16):
               self.model.current_internal_params['sigma'][j] = 0 


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
        if isinstance(action, np.ndarray):
            action = self.transpose_actions[action[0]]
        action = list(action)
        self.update_with_action(action)
        
        # Run model for jump_of steps
        model_state = [self.model_state]
        model_states = []
        for i in range(15):
            model_state = self.model.run_n_steps(model_state[-1], 1)
            model_states += model_state.tolist()
        self.model_state = model_state[-1]  # last internal state is the new current one
        self.t += self.jump_of

        # Update state
        self._update_previous_env_state()
        self._update_env_state()

        # Store history
        costs = [c.compute_cost(previous_state=np.atleast_2d(self.previous_env_state),
                                                state=np.atleast_2d(self.env_state),
                                                label_to_id=self.label_to_id,
                                                action=action,
                                                others=dict(jump_of=self.time_resolution)) for c in self.cost_function.costs]
        self.cumulative_costs[0] += costs[0]
        n_deaths = self.cost_function.compute_deaths(previous_state=np.atleast_2d(self.previous_env_state),
                                                     state=np.atleast_2d(self.env_state),
                                                     label_to_id=self.label_to_id,
                                                     action=action)

        self._update_env_state()
        self.history['actions'] += [action] * self.jump_of
        self.history['env_states'] += [self.env_state.copy()] * self.jump_of
        self.history['env_timesteps'] += list(range(self.t - self.jump_of, self.t))
        self.history['model_states'] += model_states
        self.history['vaccinations'] += [self.vaccination_politic] * self.jump_of
        self.history['deaths'] += [n_deaths] * self.jump_of#[n_deaths / self.jump_of] * self.jump_of

        # Compute cost_function
        self.history['costs'] += [costs[0] / self.jump_of for _ in range(self.jump_of)]

        cost_aggregated, costs, over_constraints = self.cost_function.compute_cost(previous_state=self.previous_env_state,
                                                                                   state=self.env_state,
                                                                                   label_to_id=self.label_to_id,
                                                                                   action=action,
                                                                                   others=dict(jump_of=self.jump_of))
        costs = costs.flatten()
        self.history['aggregated_costs'] += [cost_aggregated / self.jump_of] * self.jump_of
        self.history['costs'] += [costs / self.jump_of for _ in range(self.jump_of)]
        if self.t >= self.simulation_horizon:
            done = 1
        else:
            done = 0
        return self._normalize_env_state(self.env_state), cost_aggregated, done, dict(costs=costs)

    # Utils
    def _normalize_env_state(self, env_state):
        return (env_state / np.array(self.normalization_factors)).copy()

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

    simulation_horizon = 360
    model = get_model(model_id='heffernan_model_mg', params=dict(stochastic=False))
    cost_func = get_cost_function(cost_function_id='one_cost_death_toll', params=dict(ratio_death_to_R=0.0016))
    env = gym.make('EpidemicVaccination-v1',
                    cost_function=cost_func,
                    model=model,
                    simulation_horizon=simulation_horizon)
    env.reset()
    model_states=[]
    env.model.current_state, env.model.current_internal_params, model_states = env.initialize_model_for_vaccine()
    # Actions
    actions = ([[0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
                [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], 
                [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],  
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1], 
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], 
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1], 
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], 
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1], 
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],  
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1], 
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], 
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1], 
                [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], 
                [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],  
                [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1], 
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],   
                [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]])

    t = 0
    r = 0
    done = False
    while not done:
        out = env.step(actions[t])
        t += 1
        r += out[1]
        done = out[2]
    stats = env.unwrapped.get_data()
    # Plot
    # print(np.concatenate(np.array(modelstate), np.array(stats['history']['model_states']), axis=0))
    time = np.arange(731-370)
    # stats['history']['model_states'].pop(0)
    # print(stats['history']['aggregated_costs'])
    # print("")
    # print(stats['history']['costs'])
    plot_preds(t=np.arange(732-371),states=np.array(stats['history']['model_states']).transpose()[23], title="Évolution de la proportion d'individus vaccinés par classe d'âge avec une dose de vaccin (en %)")
    # plt.plot(np.arange(simulation_horizon),np.array(stats['history']['deaths']))
    # plt.axvline(x=0, label='Début de la campagne vaccinale', color='red', linewidth=1, linestyle='--')
    # plt.axvline(x=631-370, label='Fin de la première dose', linewidth=1, linestyle='--')
    # plt.legend()
    # plt.title("Non-cumulative cost function (number of death each month)")
    # plt.show()

    # env.model.reset()
    # model_states = []
    # for i in range(simulation_horizon):
    #     model_state = model.run_n_steps()
    #     model_states += model_state.tolist()
    # print(np.array(model_states)[0])
    
    i4tot = []
    castot = []
    for i in np.array(stats['history']['model_states']):
        tot, tat = 0, 0
        for j in i:
            tot += j[23]
            tat += j[23]+j[22]*0.5
        i4tot.append(tot)
        castot.append(tat)
    plt.plot(time, castot, label='I$_3$*0.5 + I$_4$', color='red')
    plt.plot(time, i4tot, label='I$_4$')
    # plt.plot(np.linspace(0, 200, (571-371)), (np.array(get_incidence())), label='Données SIDEP')
    plt.axvline(x=398-371, label='Lockdown 1', color='green', linewidth=1, linestyle='--')
    plt.axvline(x=546-371, label='Lockdown 2', color="purple", linewidth=1, linestyle='--')
    # plt.xlabel("Temps (en jours)")
    # plt.ylabel(r'Nombre de personnes hospitalisées')
    plt.legend()
    # plt.title("Évolution du nombre de cas incident modérés et sévères (I$_3$ + I$_4$) de COVID-19 avec vaccination (50 scénarios)")
    plt.show()