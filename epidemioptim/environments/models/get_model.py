from epidemioptim.environments.models.prague_ode_seirah_model import PragueOdeSeirahModel
from epidemioptim.environments.models.heffernan_ode_model import HeffernanOdeModel
from epidemioptim.environments.models.heffernan_ode_model_16 import HeffernanOdeModel16

list_models = ['prague_seirah', 'heffernan_model', 'heffernan_model_mg']
def get_model(model_id, params={}):
    """
    Get the epidemiological model.

    Parameters
    ----------
    model_id: str
        Model identifier.
    params: dict
        Dictionary of experiment parameters.

    """
    assert model_id in list_models, "Model id should be in " + str(list_models)
    if model_id == 'prague_seirah':
        return PragueOdeSeirahModel(**params)
    elif model_id == 'heffernan_model':
        return HeffernanOdeModel(**params)
    elif model_id == 'heffernan_model_mg':
        return HeffernanOdeModel16(**params)
    else:
        raise NotImplementedError

#TODO: add tests for model registration


