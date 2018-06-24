from src.mysocket import app
import numpy as np
from flask import Flask, render_template
from time import sleep

distribution = np.zeros(7)

@app.route('/')
def index():
    # render the template (below) that will use JavaScript to read the stream
    return render_template('index.html')

@app.route('/stream')
def stream():
    def generate():
        while 1 == 1:

            #TODO: delete
            #distribution = np.random.rand(7)
            #distribution /= distribution.sum()
            
            yield ",".join(np.char.mod('%.2f', distribution)) + '\n'
            sleep(1)
    return app.response_class(generate(), mimetype='text/plain')