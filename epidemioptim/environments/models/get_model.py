from epidemioptim.environments.models.prague_ode_seirah_model import PragueOdeSeirahModel
from epidemioptim.environments.models.heffernan_ode_model import HeffernanOdeModel

list_models = ['prague_seirah', 'heffernan_model']
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
    else:
        raise NotImplementedError

#TODO: add tests for model registration


