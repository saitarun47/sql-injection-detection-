import os
import urllib.request as request
import zipfile
from src import logger
from src.utils.utils import get_size
from src.entity.config_entity import DataIngestionConfig
from pathlib import Path
import shutil


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")



    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


    def process_file(self):
        """Handles both ZIP and non-ZIP files."""
        local_file_path = self.config.local_data_file
        unzip_dir = self.config.unzip_dir

        print(f"📂 Processing file: {local_file_path}")

        if zipfile.is_zipfile(local_file_path):
            print(" Extracting ZIP file...")
            os.makedirs(unzip_dir, exist_ok=True)
            with zipfile.ZipFile(local_file_path, "r") as zip_ref:
                zip_ref.extractall(unzip_dir)
            print(f"Extracted files: {os.listdir(unzip_dir)}")
        else:
            print("File is NOT a ZIP. Moving to target directory...")
            os.makedirs(unzip_dir, exist_ok=True)
            destination_path = os.path.join(unzip_dir, os.path.basename(local_file_path))
            shutil.move(local_file_path, destination_path)
            print(f"Moved file to {destination_path}")
            print(f" Moved file to {destination_path}")