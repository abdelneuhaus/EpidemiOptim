from epidemioptim.environments.cost_functions.costs.base_cost_function import BaseCostFunction


class DeathTollVaccine(BaseCostFunction):
    def __init__(self,
                 id_cost,
                 ratio_death_to_R,
                 scale_factor=0.65 * 1e3,
                 range_constraints=(1000, 62000)):
        """
         Economic cost computed as GDP recess due to diseased and dead people, as well as partial unemployment due to lock-downs.
         GDP is expressed in billions.

         Parameters
         ----------
         id_cost: int
             Identifier of the cost in the list of costs
         ratio_death_to_R: float
             Ratio of dead people computed from the number of recovered people, (in [0, 1]).
         scale_factor: float
             Scaling factor of the cost (in [0, 1])
         range_constraints: tuple
             Min and max values for the maximum constraint on the cost (size 2).

         Attributes
         ----------
         ratio_death_to_R
         id_cost
         """
        super().__init__(scale_factor=scale_factor,
                         range_constraints=range_constraints)
        self.id_cost = id_cost
        self.ratio_death_to_R = ratio_death_to_R

    def compute_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Computes GDP loss since the last state.

        Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        action:

        Returns
        -------
        new_deaths: 1D nd.array
            number of deaths for each state.

        """
        # compute new deaths
        popGrp = ['S1', 'S2', 'S3', 'S4', 'E21', 'E22', 'E23', 'E31', 'E32', 'E33', 'E41', 'E42', 
                  'E43', 'V11', 'V21', 'V31', 'V41', 'V12', 'V22', 'V32', 'V42', 'I2', 'I3', 'I4']
        sum_previous, sum_actual = 0.0, 0.0
        for i in range(16):
            for k in popGrp:
                sum_previous += previous_state[0][i][k]
                sum_actual += state[0][i][k]
        new_deaths = sum_previous - sum_actual
        return new_deaths

    def compute_cumulative_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Compute cumulative costs since start of episode.

        Parameters
        ----------
               Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        action:

        Returns
        -------
        cumulative_cost: 1D nd.array
            Cumulative costs for each state.
        """
        cumulative_cost = state[:, label_to_id['cumulative_cost_{}'.format(self.id_cost)]]
        print(cumulative_cost)
        return cumulative_cost
