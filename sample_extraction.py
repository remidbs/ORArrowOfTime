'''This function extracts samples following the specifications from paper
"Shuffle and Learn: Unsupervised Learning usingTemporal Order Verification" ["Misra et al.]
from the dataset used for paper "Seeing the Arrow of Time" [Pickup et al.]'''

import cv2
import numpy as np
import os

video_paths = os.listdir("img/ArrowDataAll/")
for video_path in video_paths:
    print "Starting work for video ",video_path
    imgs = []
    paths = os.listdir("img/ArrowDataAll/"+video_path)
    for path in paths:
        imgs += [cv2.imread("img/ArrowDataAll/"+video_path+"/"+path)]
    imgs_grey = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in imgs]
    ssd1 = 0
    ssd2 = 0
    it = 0
    threshold = 1545870
    w = np.zeros(len(imgs)-1)
    for t in range(len(imgs)-1):
        flow = cv2.calcOpticalFlowFarneback(imgs_grey[t],imgs_grey[t+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        w[t] = np.mean(mag)
    pb = w/np.sum(w)
    while (it < 10) & (ssd1 < threshold) & (ssd2 < threshold) :
        samples = np.ones(pb.shape)*2
        while (np.max(samples) > 1):
            samples = np.random.multinomial(5,pb)
        imgs_sampled, = np.where(samples == 1)
        ssd1 = np.sum((imgs[imgs_sampled[0]] - imgs[imgs_sampled[2]])**2)
        ssd2 = np.sum((imgs[imgs_sampled[4]] - imgs[imgs_sampled[2]])**2)
        it += 1
    if (ssd1 >= threshold) & (ssd2 >= threshold):
        print "Samples found after ",it," iterations"
        if not os.path.exists("img/Samples/"+video_path):
            os.makedirs("img/Samples/"+video_path)
        cv2.imwrite("img/Samples/"+video_path+"/"+"a.png", imgs[imgs_sampled[0]])
        cv2.imwrite("img/Samples/"+video_path+"/"+"b.png", imgs[imgs_sampled[1]])
        cv2.imwrite("img/Samples/"+video_path+"/"+"c.png", imgs[imgs_sampled[2]])
        cv2.imwrite("img/Samples/"+video_path+"/"+"d.png", imgs[imgs_sampled[3]])
        cv2.imwrite("img/Samples/"+video_path+"/"+"e.png", imgs[imgs_sampled[4]])
    else:
        print "samples not found"

    






