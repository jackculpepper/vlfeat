
import vlfeat
import Image
import numpy as np
from time import time
 
imgfname = 'testfiles/000001.pgm'
 
 
im = Image.open(imgfname)
im = np.array(im, 'f')

t0 = time()

f, d = vlfeat.vl_dsift(im, norm=True, fast=True, step=2, size=4,
                       window_size=1.5, verbose=True)

print "vl_dsift() took %.4f s"%(time() - t0)

import scipy.io
matfile = scipy.io.loadmat('testfiles/000001_pgm_dsift_float.mat')
 
dmat = matfile['d']
fmat = matfile['f']

## change from 1-based numbering to 0-based
fmat[:2,:] -= 1

print "d:"
print d
print "dmat:"
print dmat
 
diff = dmat - d
mse_desc = (diff**2).sum(0).mean()
print "avg desc error:", mse_desc
 
diff = fmat - f
mse_frame = (diff**2).sum(0).mean()
print "avg frame error:", mse_frame

assert(mse_desc < 1e-4)
assert(mse_frame < 1e-4)

