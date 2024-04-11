from landcoverpy.execution_mode import ExecutionMode
from landcoverpy.workflow import workflow

from landcoverpy.model_training import train_model_land_cover, train_dnn_model_land_cover
from landcoverpy.utilities.aoi_tiles import get_list_of_tiles_in_andalucia


land_cover_dataset = "dataset_postprocessed.csv"



#train_model_land_cover(land_cover_dataset, n_jobs = 1)
#train_dnn_model_land_cover(land_cover_dataset, n_jobs = 1)

workflow(execution_mode=ExecutionMode.LAND_COVER_PREDICTION, tiles_to_predict=get_list_of_tiles_in_andalucia())