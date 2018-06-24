

import cv2
import numpy as np

def get_video_frames(queue, frame_num=30, out_shape=(48,48)):
    """Print the camera output into the out object"""
    frame_count = 0
    camera = cv2.VideoCapture(0)
    while True:
        frame_count += 1
        return_value,image = camera.read()
        flip = cv2.flip(image, 1)
        gray = cv2.cvtColor(flip,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',gray)
        if(frame_count>frame_num):
            frame_count = 0
            resized = cv2.resize(gray, (48,48))
            resized = np.array(resized)
            queue.put(resized)
            # give gray to the NN
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    out = np.array(np.zeros((48,48)))
    get_video_frames(out)
