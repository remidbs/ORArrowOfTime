import numpy as np
import os
from scipy.misc import imread, imsave

video_paths = os.listdir("Samples_resized/")
for video_path in video_paths:

    paths = os.listdir("Samples_resized/"+video_path)
    imgs = []
    for path in paths:
        imgs += [imread("Samples_resized/"+video_path+"/"+path)]

    names = ["a.png","b.png","c.png","d.png","e.png"]

    # Identical
    i_name = 0
    for img in imgs:
        if not os.path.exists("Samples_resized_extended/" + video_path + "_ii/"):
            os.makedirs("Samples_resized_extended/" + video_path + "_ii/")
        imsave("Samples_resized_extended/" + video_path + "_ii/" + names[i_name], img)
        i_name+=1

    # Time mirror
    i_name = 0
    for img in reversed(imgs):
        if not os.path.exists("Samples_resized_extended/" + video_path + "_im/"):
            os.makedirs("Samples_resized_extended/" + video_path + "_im/")
        imsave("Samples_resized_extended/" + video_path + "_im/" + names[i_name], img)
        i_name += 1

    # Spatial mirror
    i_name = 0
    for img in imgs:
        if not os.path.exists("Samples_resized_extended/" + video_path + "_mi/"):
            os.makedirs("Samples_resized_extended/" + video_path + "_mi/")
        imsave("Samples_resized_extended/" + video_path + "_mi/" + names[i_name], np.fliplr(img))
        i_name += 1

    # Spatial mirror + Time mirror
    i_name = 0
    for img in reversed(imgs):
        if not os.path.exists("Samples_resized_extended/" + video_path + "_mm/"):
            os.makedirs("Samples_resized_extended/" + video_path + "_mm/")
        imsave("Samples_resized_extended/" + video_path + "_mm/" + names[i_name], np.fliplr(img))
        i_name += 1

