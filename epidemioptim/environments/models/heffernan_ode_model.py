from odeintw import odeintw

from epidemioptim.environments.models.base_model import BaseModel
from epidemioptim.utils import *

PATH_TO_FITTED_PARAMS = get_repo_path() + '/data/model_data/estimatedIndividualParameters.csv'
PATH_TO_FITTED_COV = get_repo_path() + '/data/model_data/data_cov.csv'

a=pd.read_csv(PATH_TO_FITTED_PARAMS)
print(a)