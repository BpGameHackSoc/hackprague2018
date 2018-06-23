import os
import csv
import argparse
import numpy as np 
import scipy.misc

file = 'data/fer2013/fer2013.csv'
output = 'data/fer2013/extracted'

w, h = 48, 48
image = np.zeros((h, w), dtype=np.uint8)
id = 1
with open(file, 'r') as csvfile:
    datareader = csv.reader(csvfile, delimiter =',')
#     print headers
    for i,row in enumerate(datareader):
        if i==0:
            continue
        emotion = row[0]
        pixels = list(map(int, row[1].split()))
#         print(row)
        usage = row[2]
        #print emotion, type(pixels[0]), usage
        pixels_array = np.asarray(pixels)

        image = pixels_array.reshape(w, h)
        #print image.shape

        stacked_image = np.dstack((image,) * 3)
        #print stacked_image.shape


        image_folder = os.path.join(output, usage)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_file =  os.path.join(image_folder , str(id) + '.jpg')
        scipy.misc.imsave(image_file, stacked_image)
        id += 1
        if id % 100 == 0:
            print('Processed {} images'.format(id))

print("Finished processing {} images".format(id))
