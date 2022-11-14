from math import sqrt
import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        #assumption: batch_size must be smaller than epoch_size! 
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        #current epoch number
        self.epoch_nr = -1
        #current image number
        self.image_nr = 0
        #only read once
        self.image_dir = os.listdir(self.file_path)
        self.image_count = len(self.image_dir)
        self.file_dict = {}
        with open(self.label_path) as json_file:
            self.label_dict = json.load(json_file)
            
        random.seed(10)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        images = []
        labels = []
        epoch_changed=False
        
        for i in range (0, self.batch_size):
            self.image_nr += 1
            if (epoch_changed):
                self.epoch_nr += 1
                epoch_changed = False
            if (len(self.file_dict) == 0):
                epoch_changed = True
                self.image_nr = 0
                if (self.shuffle):
                    random.shuffle(self.image_dir)
                for i in range(0, self.image_count):
                    self.file_dict[i] = self.image_dir[i]
            
            img = np.load(os.path.join(self.file_path, self.file_dict[self.image_nr]))
            img = self.augment(img)
            images.append(img)
            labels.append(self.label_dict[self.file_dict[self.image_nr][0:-4]])
            self.file_dict.pop(self.image_nr, None)
            
        return np.asarray(images), np.asarray(labels)
            


    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if (self.mirroring):
            random_axis = random.randint(0, 1)
            img = np.flip(img, axis=random_axis)
            
        if (self.rotation):
            rotations = [1,2,3]
            img = np.rot90(img, random.choice(rotations))
            
        if self.image_size != None:
            img = skimage.transform.resize(img, self.image_size)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_nr       
    

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    
    def show(self):
        images, labels = self.next()
        fig = plt.figure()
        dim = int(sqrt(len(images))) + 1
        
        for i in range(0, len(images)):
            fig.add_subplot(dim, dim, i+1)
            plt.imshow(images[i])
            plt.title(self.class_name(labels[i]))
        
        plt.show()

