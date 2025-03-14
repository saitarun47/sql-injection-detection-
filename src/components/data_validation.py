import os
from src import logger
import pandas as pd
from src.entity.config_entity import DataValidationConfig

class DataValidation:  
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            data = pd.read_csv(self.config.unzip_data_dir, sep=",", engine="python", on_bad_lines="skip",encoding='utf-16')
            data.columns = data.columns.str.strip()  

            all_cols = set(data.columns)
            expected_cols = set(self.config.all_schema.keys())

           
            missing_cols = expected_cols - all_cols
            extra_cols = all_cols - expected_cols

            if missing_cols or extra_cols:
                validation_status = False
                logger.error(f"Schema validation failed!")
                logger.error(f"Missing Columns: {missing_cols}")
                logger.error(f"Extra Columns: {extra_cols}")
            else:
                validation_status = True
                logger.info("Schema validation passed!")

          
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            raise e
