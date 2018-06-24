import sys
sys.path.append('..')
import threading, queue
import getdata
import mobile
import keras
import numpy as np

def scale_frame(frame):
    frame = frame.astype(np.float32) / 255.
    return frame

def call_js(y):
    print(y)

def main():
    model = keras.models.load_model('models/hg_first_53', custom_objects={'relu6': mobile.relu6})
    q = queue.Queue()
    t = threading.Thread(target=getdata.get_video_frames, kwargs={'queue':q,'frame_num':1},daemon=True)
    t.start()
    while True:
        raw_frame = q.get(block=True)
        frame = scale_frame(raw_frame)
        frame_model_comp = np.expand_dims(np.expand_dims(frame,0),-1)
        res = list(model.predict(frame_model_comp)[0])
        call_js(res)

if __name__ == '__main__':
    main()


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