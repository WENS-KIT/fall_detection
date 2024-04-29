#!/usr/bin/python3

import time,board,busio
import numpy as np
import adafruit_mlx90640
import datetime as dt
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000) # setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c) # begin MLX90640 with I2C comm
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ # set refresh rate

mlx_shape = (24,32)

print ('---')

frame = np.zeros((24*32,)) # setup array for storing all 768 temperatures
def plot_update():
    fig.canvas.restore_region(ax_background) # restore background
    mlx.getFrame(frame) # read mlx90640
    data_array = np.fliplr(np.reshape(frame,mlx_shape)) # reshape, flip data
    data_array = ndimage.interpolation.zoom(data_array,zoom=10) # interpolate
    therm1.set_array(data_array) # set data
    therm1.set_clim(vmin=np.min(data_array),vmax=np.max(data_array)) # set bounds
    cbar.on_mappable_changed(therm1) # update colorbar range

    ax.draw_artist(therm1) # draw new thermal image
    fig.canvas.blit(ax.bbox) # draw background
    fig.canvas.flush_events() # show the new image
    return

Tmax = 35
Tmin = 30

def td_to_image(f):
    norm = np.uint8((f + 40)*6.4)
    norm.shape = (24,32)
    return norm

time.sleep(4)

t0 = time.time()

################BackgroundSubstraction######################
##fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100)
############################################################
try:
    #out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 10.0,(640, 480))

    while True:
        # waiting for data frame
        mlx.getFrame(frame) # read MLX temperatures into frame var
        img16 = (np.reshape(frame,mlx_shape)) # reshape to 24x32 
        #img16 = (np.fliplr(img16))
        
        ta_img = td_to_image(img16)
        # Image processing
        img = cv2.applyColorMap(ta_img, cv2.COLORMAP_JET)
        img = cv2.resize(img, (640,480), interpolation = cv2.INTER_CUBIC)
        img = cv2.flip(img, 1)
        
        #text = 'Tmin = {:+.1f} Tmax = {:+.1f} FPS = {:.2f}' .format(frame.min(), frame.max(), 1/(time.time() - t0))
        #img = cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)      
        
        #median Filter
        #img_median = img
        #k_size = 25
        #median = cv2.medianBlur(img_median, k_size)

        #gaussian Filter
        #img_gaussian = img
        #gaussian = cv2.GaussianBlur(img_gaussian,(9,9),cv2.BORDER_DEFAULT)

        #Bilateral Filtering
        img_bilateral = img
        bilateral = cv2.bilateralFilter(img_bilateral,9,75,75)
        
        
######################BackgroundSubstraction#########################################
#        fgmask = fgbg.apply(median)
#        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
#
#        for index, centroid in enumerate(centroids):
#          if stats[index][0] == 0 and stats[index][1] == 0:
#            continue
#          if np.any(np.isnan(centroid)):
#            continue
#
#          x, y, width, height, area = stats[index]
#          centerX, centerY = int(centroid[0]), int(centroid[1])
#
#          if area > 100:
#            cv2.circle(erosion, (centerX, centerY), 1, (0, 255, 0), 2)
#            cv2.rectangle(erosion, (x,y), (x + width, y + height), (0, 0, 255))
#
#
#####################################################################################

        #Erosion
        #kernel_erosion = np.ones((3,3), np.uint8)
        #erosion = cv2.erode(fgmask, kernel_erosion, iterations = 1)
        
        #Dilation
        #kernel_dilation = np.ones((5,5), np.uint8)
        #dilation = cv2.dilate(fgmask, kernel_dilation, iterations = 3)

        print ('--imshow')

        cv2.imshow('Output1', img)
        #cv2.imshow('Output2', median)
        #cv2.imshow('Output3', gaussian)
        cv2.imshow('Output4', bilateral)
        #cv2.imshow('Output5', fgmask)
        #cv2.imshow('Output6', erosion)
        #cv2.imshow('Output7', dilation)

        #out.write(erosion)
        key = cv2.waitKey(1) & 0xFF

        t0 = time.time()



except KeyboardInterrupt:
    # to terminate the cycle
    cv2.destroyAllWindows()
    print(' Stopped')

# just in case
cv2.destroyAllWindows()
