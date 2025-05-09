from dotenv import load_dotenv
import os

load_dotenv()

class Env:
    def __init__(self):
        self.FILE_ID = os.getenv("FILE_ID")

env = Env()
    
