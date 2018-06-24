from flask import Flask
app = Flask(__name__)
from src.socket import routes