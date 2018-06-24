from . import app
import numpy as np
from flask import Flask, render_template
from time import sleep
# import sys
from src import runner




@app.route('/')
def index():
    # render the template (below) that will use JavaScript to read the stream
    return render_template('index.html')

@app.route('/stream')
def stream():
    def generate():
        callback = runner.main()
        while 1 == 1:

            #TODO: delete
            #distribution = np.random.rand(7)
            #distribution /= distribution.sum()
            nums =  callback.__next__()
            print(nums)
            yield ",".join(np.char.mod('%.2f', nums)) + '\n'
            sleep(1)
    return app.response_class(generate(), mimetype='text/plain')