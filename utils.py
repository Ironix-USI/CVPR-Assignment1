import time
import numpy as np
from numpy.linalg import svd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()
    
    
def select_points(img, N_points):
    tellme('Click to begin')
    plt.imshow(img)

    plt.waitforbuttonpress()
    pts = []

    while len(pts) < N_points:
        tellme('Select {} points'.format(N_points))
        pts = np.asarray(plt.ginput(N_points, timeout=-1))
        if len(pts) < N_points:
            tellme('Too few points, starting over')
            time.sleep(1)  # Wait a second
                
        tellme('Key click to confirm')

        if plt.waitforbuttonpress():
            break

    plt.close()
    return pts


def draw_line(point1,point2, options = 'b'):
    
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    
    plt.plot(x_values, y_values, options)


def bilinear_interpolate(im, xx, yy):
    output = []
    
    for x,y in zip(xx,yy):

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1]-1)
        x1 = np.clip(x1, 0, im.shape[1]-1)
        y0 = np.clip(y0, 0, im.shape[0]-1)
        y1 = np.clip(y1, 0, im.shape[0]-1)

        Ia = im[y0,x0]
        Ib = im[y1,x0]
        Ic = im[y0,x1]
        Id = im[y1,x1]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)
        
        output.append(wa*Ia + wb*Ib + wc*Ic + wd*Id)

    return np.asarray(output,dtype=int)


def imwarpLinear(I,H, bb, step = 100):

    #1. create a grid using the four points bb

    x = np.linspace(bb[0],bb[2],step)
    y = np.linspace(bb[1],bb[3],step)
    xx, yy = np.meshgrid(x, y)

    z = np.ones(step*step)
    grid = np.matrix([xx.flatten(),yy.flatten(),z])
    
    #2. Apply the (inverse) transformation to every point of the grid

    transformed_grid = (np.linalg.inv(H) @ grid).getA()
    
    # REMEMBER NORMALIZATION
    xx = np.divide(transformed_grid[0],transformed_grid[2])
    yy = np.divide(transformed_grid[1],transformed_grid[2])


    plt.scatter(xx,yy)
    plt.imshow(I)
    plt.show()
    #3.interpolate the points to obtain the values and then plot them on a regular grid
    # you can cheat by using the `function interpolate.interp2d`

    tranformed = bilinear_interpolate(I,xx.flatten(),yy.flatten())
    
    return tranformed.reshape(step,step,3)



def create_A(x, xP):
    
    A = np.array([
        np.hstack((np.array([0, 0, 0]), -xP[0][2]*np.transpose(x[0]), xP[0][1]*np.transpose(x[0]))),
        np.hstack((xP[0][2]*np.transpose(x[0]), np.array([0, 0, 0]), -xP[0][0]*np.transpose(x[0]))),
        
        np.hstack((np.array([0, 0, 0]), -xP[1][2]*np.transpose(x[1]), xP[1][1]*np.transpose(x[1]))),
        np.hstack((xP[1][2]*np.transpose(x[1]), np.array([0, 0, 0]), -xP[1][0]*np.transpose(x[1]))),
        
        np.hstack((np.array([0, 0, 0]), -xP[2][2]*np.transpose(x[2]), xP[2][1]*np.transpose(x[2]))),
        np.hstack((xP[2][2]*np.transpose(x[2]), np.array([0, 0, 0]), -xP[2][0]*np.transpose(x[2]))),
        
        np.hstack((np.array([0, 0, 0]), -xP[3][2]*np.transpose(x[3]), xP[3][1]*np.transpose(x[3]))),
        np.hstack((xP[3][2]*np.transpose(x[3]), np.array([0, 0, 0]), -xP[3][0]*np.transpose(x[3])))
        ])
        
        
    return A


def homographyEstimation(x, xP):
    
    A = create_A(x, xP)
    U,D,V = svd(A)
    H = (V[8]/V[8][8]).reshape((3,3))  
    
    return H



