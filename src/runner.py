import sys
sys.path.append('..')
import threading, queue
try:
    from . import getdata
    from . import mobile
except ImportError:
    import getdata
    import mobile
import keras
import numpy as np
import time
from PIL import Image
# from mysocket import app


labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def scale_frame(frame):
    im = Image.fromarray(np.array(frame))
    im = im.resize((48,48))
    frame = np.array(im).astype(np.float32)
    return frame

def call_js(y):
    # import src.mysocket.routes as http_requester
    app.distribution = y

def show(dist,queue_size=None,sleep_time=None):
    dist = dist*100
    print(dist)
    print("The person is: {}".format(labels[np.argmax(dist)]))
    if queue_size is not None and sleep_time is not None:
        print("The queue size is {} while the sleep time was {}".format(queue_size,sleep_time))



def main():
    model = keras.models.load_model('models/hg_first_53', custom_objects={'relu6': mobile.relu6})
    q = queue.LifoQueue()
    t = threading.Thread(target=getdata.get_video_frames, kwargs={'queue':q,'frame_num':1},daemon=True)
    t.start()
    while True:
        raw_frame = q.get(block=True)
        frame = scale_frame(raw_frame)
        frame_model_comp = np.expand_dims(np.expand_dims(frame,0),-1)
        res = (model.predict(frame_model_comp)[0])
        # show(res,q.qsize(),2)
        # call_js(res)
        yield res
        if not q.empty():
            q.queue.clear()

if __name__ == '__main__':
    for x in main():
        show(x)


#
# def worker():
#     while True:
#         item = q.get()
#         do_work(item)
#         q.task_done()
#
# q = Queue()
# for i in range(num_worker_threads):
#      t = Thread(target=getdata.get_video_frames,kwargs=(queue=))
#      t.daemon = True
#      t.start()
#
# for item in source():
#     q.put(item)