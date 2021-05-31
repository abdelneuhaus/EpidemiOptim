from epidemioptim.environments.cost_functions.one_cost_death import OneCostDeathToll
from epidemioptim.environments.cost_functions.multi_cost_death_gdp_controllable import MultiCostDeathGdpControllable
from epidemioptim.environments.cost_functions.costs.death_toll_cost_vaccine import DeathTollVaccine

def get_cost_function(cost_function_id, params={}):
    if cost_function_id == 'multi_cost_death_gdp_controllable':
        return MultiCostDeathGdpControllable(**params)
    elif cost_function_id == 'one_cost_death_toll':
        return OneCostDeathToll(**params)
    elif cost_function_id == 'death_toll_cost_vaccine':
        return DeathTollVaccine(**params)
    else:
        raise NotImplementedError
