from flask import Flask
app = Flask(__name__)
from . import routes
import numpy as np
# app.distribution = np.zeros(7)