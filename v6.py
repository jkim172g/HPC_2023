from matplotlib import pyplot
import numpy as np
from timeit import default_timer as timer
from numba import cuda
from numba import *

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

@jit
def collision():
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
        Feq[:, :, i] = rho * w* (
            1 + 3 * (cx*ux + cy*uy) + 9 * (cx*ux + cy*uy)**2 / 2 - 3 *(ux**2 + uy**2)/2
        )

    return F + -(1/tau) * (F-Feq)

@cuda.jit
def collision_kernel():
    F = collision_gpu



collision_gpu = cuda.jit(device=True)(collision)


plot_every = 50



Nx = 400
Ny = 100
tau = .53
Nt = 30000

#lattice speeds and weights
NL = 9
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36,1 /9, 1/36])

#initial conditions
F = np.ones((Ny, Nx, NL)) + 0.1 * np.random.randn(Ny, Nx, NL)
F[:, :, 3] = 2.3

cylinder = np.full((Ny, Nx), False)

for y in range(Ny):
    for x in range(Nx):
        if(distance(Nx / 4, Ny / 2, x, y) < 13):
            cylinder[y][x] = True

blockdim = (32, 8)
griddim = (32,16)

            
#def main():     
start = timer()

#main loop
for it in range(Nt):
    for i, cx, cy in zip(range(NL), cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)

    bndryF = F[cylinder, :]
    bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

    rho = np.sum(F, 2)
    ux = np.sum(F * cxs, 2) / rho
    uy = np.sum(F * cys, 2) / rho


    F[cylinder, :] = bndryF
    ux[cylinder] = 0
    uy[cylinder] = 0

    #collision
    gpu_F = cuda.to_device(F)
    collision_kernel[griddim, blockdim]()
    gpu_F.copy_to_host()

    if(it%plot_every == 0):
        filename = 'v6_frames/frame' + str(int(it/plot_every)) + '.png'
        pyplot.imsave(filename, np.sqrt(ux**2+uy**2))
        #pyplot.cla()


dt = timer() - start

print("%d iterations in %f seconds" % (Nt, dt))
        
        

        
#if __name__ == "__main__":
#    main()



