from fastapi import FastAPI, HTTPException
from deepface import DeepFace
from typing import List
import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

app = FastAPI()

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "asctime": self.formatTime(record, self.datefmt),
            "name": record.name,
            "levelname": record.levelname,
            "message": record.getMessage(),
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "threadName": record.threadName,
        }
        return json.dumps(log_entry)
    
# Configure logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the root logger level

    file_handler = RotatingFileHandler(
        'log.json',
        maxBytes=10**6,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JsonFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(JsonFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

setup_logging()

@app.get("/")
async def root():
    return {"message": "Server is running..."}

@app.get("/match", tags=["Face Match"])
async def isMatch(input_url: str, check_url: str):
    """
    This function checks if two pictures are the same person
    """
    result = DeepFace.verify(
      img1_path = input_url,
      img2_path = check_url,
      model_name = models[5],
    )
    print(result)
    return {"result": result}

# Function to load URLs from a JSON file
def load_urls():
    with open("files.json", "r") as file:  # Adjust the path to where your JSON file is stored
        data = json.load(file)
    return data["images"]

@app.get("/find", tags=["Database Searching"])
def match(input:str):
    """
    This function verifies if a face exists in the database
    """
    urls = load_urls()
    for image in urls:
      result = DeepFace.verify(
      img1_path = input,
      img2_path = image,
      model_name = models[5])
      logging.info(f"Checking match on {image}...")
      if(result['verified']):
          return {"message": f"Person Found {image}"}
      logging.info("match not found")
    return {"message": "Processed all URLs, person not found"}


@app.get("/images")
def list_images():
    """
    Gets all the images in the database as an a json
    """
    urls = load_urls()
    return {"images": urls }


@app.get("/tests")
def run_tests():
    for model in models:
        for root, dirs, files in os.walk('testing'):
            for test_image in files:
                logging.info(f"Processing: model-{model} file-{test_image}")
                for root,dirs, files in os.walk('faces'):
                    for db_image in files:
                        try:
                            # Measure inference time
                            start_time = time.time()
                            result = DeepFace.verify(f"testing/{test_image}", f"faces/{db_image}", model_name = model)
                            inf_time = time.time() - start_time
                            # Add additional information
                            extended_result = {
                                "test_image": test_image,
                                "db_image": db_image,
                                "inf_time": inf_time,
                                **result
                            }
                            logging.info(extended_result)
                        except Exception as e:
                            print(f"Error processing {test_image} with {db_image}: {e}")
    '''
    result = DeepFace.verify("/home/muaaz/University/Hackathon/ImageMatch/faces/60269024.jpg","/home/muaaz/University/Hackathon/ImageMatch/faces/60269024.jpg")
    print(result)
    '''

def rename_files_in_directory(directory: str, suffix: str):
    try:
        # Validate directory existence
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")
        
        # Iterate over files in the directory
        for filename in os.listdir(directory):
            old_file_path = os.path.join(directory, filename)
            if os.path.isfile(old_file_path):
                # Construct the new file name
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}-{suffix}{ext}"
                new_file_path = os.path.join(directory, new_filename)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
        
        return {"message": "Files renamed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/rename-files/")
def rename_files(dir: str, suffix: str):
    return rename_files_in_directory(dir, suffix)
    
"""
Testing and model evaluations:

Models:
1. x
2. y
3. z

Testing Data:
16 - Innocent People (False)
8 - Criminals Std
8 - Criminals variance

For each model, 
  For each image in testing, we will search for a match
    if it correctly matched, we will score it.

"""