from gym.envs.registration import register
from epidemioptim.environments.gym_envs.get_env import get_env

register(id='EpidemicDiscrete-v0',
         entry_point='epidemioptim.environments.gym_envs.epidemic_discrete:EpidemicDiscrete')
register(id='EpidemicVaccination-v0',
         entry_point='epidemioptim.environments.gym_envs.epidemic_vaccination:EpidemicVaccination')
register(id='EpidemicVaccination-v1',
         entry_point='epidemioptim.environments.gym_envs.epidemic_vaccination_16:EpidemicVaccinationMultiGroups')