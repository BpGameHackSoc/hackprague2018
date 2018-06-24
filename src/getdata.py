import cv2
import numpy as np
import queue

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def apply_offsets(face_coordinates, offsets, max_width=640, max_height=480):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (max(x - x_off,0), min(x + width + x_off,max_width),
            max(y - y_off,0), min(y + height + y_off,max_height))

def get_video_frames(queue, frame_num=10, out_shape=(48,48)):
    """Print the camera output into the out object"""
    frame_count = 0

    detection_model = cv2.CascadeClassifier("src/detection_models/haarcascade_frontalface_default.xml")

    camera = cv2.VideoCapture(0)
    face = None
    printed_value = None
    while True:
        frame_count += 1
        return_value,image = camera.read()
        #image = np.array(image)
        #crop = image[:,80:-80]
        flip = cv2.flip(image, 1)
        gray = cv2.cvtColor(flip,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('image',gray)
        if(frame_count>frame_num):
                
            has_face = False
            faces = detect_faces(detection_model, gray)
            face_img = -1
            for new_face in faces:
                face = new_face
                has_face = True
                


            if(has_face):
                #print(face_img.size)
                #cv2.imshow('image',face_img)
                pass
            frame_count = 0
            resized = cv2.resize(gray, (48,48))
            resized = np.array(resized)

            # give gray to the NN
        
        try:
            cv2.rectangle(gray,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0),1)
            
            face_img = gray[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
            cv2.imshow('image',face_img)
            printed_value = face_img
        except:
            cv2.imshow('image',gray)
            printed_value = gray
            continue
        queue.put(printed_value)

        #cv2.imshow('image',gray)

        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    out = queue.Queue()
    get_video_frames(out)
