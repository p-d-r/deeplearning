import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        assert(resolution%tile_size == 0)
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None
        
        
    def draw(self):
        assert(self.resolution%2*self.tile_size == 0)
        #construct a single tile of dimension tile_size*tile_size
        tile_even = np.concatenate((np.zeros((self.tile_size, self.tile_size)), np.ones((self.tile_size, self.tile_size))), axis=1)
        tile_uneven = np.concatenate((np.ones((self.tile_size, self.tile_size)), np.zeros((self.tile_size, self.tile_size))), axis=1)
        tile = np.vstack((tile_even, tile_uneven))
        #repeat the tile resolution/tile_size/2 times along both axes
        self.output = np.tile(tile, ((int)(self.resolution/self.tile_size/2), (int)(self.resolution/self.tile_size/2)))
        return self.output.copy()
    
        
    def show(self):
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.show()
        
        


class Circle: 
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
        
    
    def draw(self):
        x = np.arange(0, self.resolution, 1)
        y = np.arange(0, self.resolution, 1)
        xx, yy = np.meshgrid(x, y)
        #circle as geometry: x^2 + y^2 < radius^2 (if the center is at (0,0))
        #translate from self.position to (0,0), compute circle inliers
        self.output = (((xx-self.position[0])**2 + (yy-self.position[1])**2) <= self.radius**2) 
        return self.output.copy()
        
        
    def show(self):
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.show()
        
        

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None
        
        
    def draw(self):
        #blue is max at left side, and min at right side
        blue_plane = np.tile(np.linspace(1, 0, self.resolution), (self.resolution, 1))
        #red is max at right side and min at left side
        red_plane = np.tile(np.linspace(0, 1, self.resolution), (self.resolution, 1))
        #green is max at bottom and min at top
        green_plane = np.tile(np.linspace(0, 1, self.resolution).reshape(self.resolution, 1), (1, self.resolution))
        self.output = np.stack([red_plane, green_plane, blue_plane], axis = 2)
        return self.output.copy()
    
    
    def show(self):
        plt.imshow(self.output)
        plt.show()
        