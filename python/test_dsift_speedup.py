
import vlfeat
import Image
import os
import numpy as np
from time import time
 
imgfname = 'testfiles/000001.pgm'
 
 
im = Image.open(imgfname)
im = np.array(im, 'f')

t0 = time()

os.environ['OMP_NUM_THREADS'] = '8'
f, d = vlfeat.vl_dsift(im, norm=True, fast=True, step=1, size=4,
                       window_size=1.5, verbose=True)
time_fast = time() - t0
print "vl_dsift() took %.4f s with 8 threads" % time_fast


os.environ['OMP_NUM_THREADS'] = '1'
f, d = vlfeat.vl_dsift(im, norm=True, fast=True, step=1, size=4,
                       window_size=1.5, verbose=True)
time_slow = time() - t0
print "vl_dsift() took %.4f s with 1 thread" % time_slow

speedup = time_slow / time_fast
print "speedup = %.4f" % speedup

## produce desired output and save it
#np.savez('testfiles/000001_pgm_dsift_float_speedup.npz', f, d)

df = np.load('testfiles/000001_pgm_dsift_float_speedup.npz')

fmat = df['arr_0']
dmat = df['arr_1']

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

