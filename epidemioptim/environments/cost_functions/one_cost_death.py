from epidemioptim.environments.cost_functions.costs.death_toll_cost_vaccine import DeathTollVaccine
from epidemioptim.environments.cost_functions.base_multi_cost_function import BaseMultiCostFunction
import numpy as np


class OneCostDeathToll(BaseMultiCostFunction):
    def __init__(self,
                 ratio_death_to_R=0.0001,
                 use_constraints=False
                 ):
        """
        Multi-objective cost functions with two costs: death toll and gdp recess. It is controllable by three parameters:
        the mixing parameter beta, and one constraints of maximum cumulative cost for each of them.

        Parameters
        ----------
        ratio_death_to_R: float
            Ratio of deaths over recovered individuals (in [0, 1]).
        use_constraints: bool
            Whether to use constraints on the maximum values of cumulative rewards.
        """
        super().__init__(use_constraints=use_constraints)

        # Initialize cost functions
        self.death_toll_cost = DeathTollVaccine(id_cost=0, ratio_death_to_R=ratio_death_to_R)
        self.costs = [self.death_toll_cost]
        self.nb_costs = len(self.costs)
        self.goal_dim = 1
        self.constraints_ids = []
        self.action = []

    def sample_goal_params(self):
        """
        Sample goal parameters.

        Returns
        -------
        goal: 1D nd.array
            Made of three params in [0, 1]: beta is the mixing parameter,
            the following are normalized constraints on the maximal values of cumulative costs.

        """
        action = [0,0,0]
        return np.array(action)

    def get_eval_goals(self, n):
        goals = []
        eval_goals = np.atleast_2d(np.array([0] * n + [0.25] * n + [0.5] * n + [0.75] * n + [1] * n)).transpose()
        return eval_goals


    def get_main_goal(self):
        eval_goals = np.array([0.5])
        return eval_goals

    def set_goal_params(self, goal):
        """
        Set a goal.

        Parameters
        ----------
        goal: 1D nd.array
            Should be of size 3: mixing parameter beta and normalized constraints, all in [0, 1].

        """
        self.beta = goal[0]
        if self.use_constraints:
            if len(goal[1:]) == self.nb_costs:
                for v, c in zip(goal[1:], self.costs):
                    c.set_constraint(v)
            else:
                for c in self.costs:
                    c.set_constraint(1.)


    def compute_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Compute each cost and an aggregated measure of costs as well as constraints.

        Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        action: int or nd.array
            Int for discrete envs and nd.array in continuous envs.

        Returns
        -------
        cost_aggregated: float
            Aggregated cost using the beta mixing parameter.
        costs: list of floats
            All costs.
        over_constraints: list of bools
            Whether the constraints are violated.
        """
        previous_state = np.atleast_2d(previous_state)
        state = np.atleast_2d(state)

        # compute costs
        costs = np.array([c.compute_cost(previous_state, state, label_to_id, action, others) for c in self.costs]).transpose()
        cumulative_costs = np.array([c.compute_cumulative_cost(previous_state, state, label_to_id, action, others) for c in self.costs]).transpose()

        # Apply constraints
        cost_aggregated = costs[0]#self.compute_aggregated_cost(costs.copy())
        over_constraints = 0#np.atleast_2d([False] * state.shape[0]).transpose()
        return cost_aggregated, costs, over_constraints

    def compute_deaths(self, previous_state, state, label_to_id, action, others={}):
        """
        Compute death toll

       Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        action: int or nd.array
            Int for discrete envs and nd.array in continuous envs.

        Returns
        -------
        int
            Number of deaths
        """
        return self.costs[0].compute_cost(previous_state, state, label_to_id, action, others)

    def compute_aggregated_cost(self, costs, beta=None, constraints=None):
        """
        Compute aggregated measure of cost with mixing beta and optional constraints.
        Parameters
        ----------
        costs: 2D nd.array
            Array of costs (n_points, n_costs).
        beta: float
            Mixing parameters for the two costs.
        constraints: nd.array of bools, optional
            Whether constraints are violated (n_point, n_costs).

        Returns
        -------
        float
            Aggregated cost.
        """
        factors = np.array([1 - beta, beta])
        normalized_costs = np.array([cf.scale(c) for (cf, c) in zip(self.costs, costs.transpose())])
        cost_aggregated = np.matmul(factors, normalized_costs)

        if self.use_constraints:
            if constraints is not None:
                cost_aggregated[np.argwhere(np.sum(constraints, axis=1) > 0)] = 100
        return cost_aggregated
