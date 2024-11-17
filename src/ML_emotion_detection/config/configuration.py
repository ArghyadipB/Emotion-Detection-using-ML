# Goes to src/configuration.py
from src.ML_emotion_detection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.ML_emotion_detection.utils.common import read_yaml, create_directories  
from src.ML_emotion_detection import logger
from src.ML_emotion_detection.entity.config_entity import DataIngestionConfig, DataValidationConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        logger.info(f'Reading CONFIG YAML FROM {config_filepath}')
        self.config = read_yaml(config_filepath)
        
        logger.info(f'Reading PARAMS YAML FROM {params_filepath}')
        self.params = read_yaml(params_filepath)
        
        logger.info(f'Reading SCHEMA YAML FROM {schema_filepath}')
        self.schema = read_yaml(schema_filepath)

        logger.info(f'Creating directory stored at artifacts_root of CONFIG YAML')
        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        logger.info(f'Creating directory stored at root_dir of CONFIG YAML')
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config