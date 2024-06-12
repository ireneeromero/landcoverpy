from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.workflow import workflow_DNN
from landcoverpy.workflow_RF import workflow_RF

from landcoverpy.model_training import train_model_land_cover, train_dnn_model_land_cover
from landcoverpy.utilities.aoi_tiles import get_list_of_tiles_in_andalucia


# To train workflow using DNN models
#train_dnn_model_land_cover("base", n_jobs = 1) # Using basic model
train_dnn_model_land_cover("otro", n_jobs = 1)

# To train workflow using RF models
#train_model_land_cover(land_cover_dataset, n_jobs = 1)



# To predict workflow using DNN models
#workflow_DNN(execution_mode=ExecutionMode.LAND_COVER_PREDICTION, tiles_to_predict=get_list_of_tiles_in_andalucia())

# To predict workflow using RF models
#workflow_RF(execution_mode=ExecutionMode.LAND_COVER_PREDICTION, tiles_to_predict=get_list_of_tiles_in_andalucia())
