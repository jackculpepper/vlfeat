
import vlfeat
import Image
import numpy as np
from time import time
 
imgfname = 'testfiles/000001.jpg'
 
 
im = Image.open(imgfname)
im = np.array(im, 'f')

t0 = time()

f, d = vlfeat.vl_phow_color(im, color='rgb')

print "vl_phow_color() took %.4f s"%(time() - t0)

t0 = time()

with open('testfiles/d_jpg_phow_float_color=rgb.npy', 'w') as fh:
    np.save(fh, d)

with open('testfiles/f_jpg_phow_float_color=rgb.npy', 'w') as fh:
    np.save(fh, f)

print "save took %.4f s"%(time() - t0)

import scipy.io
matfile = scipy.io.loadmat('testfiles/000001_jpg_phow_float_color=rgb.mat')
 
dmat = matfile['d']
fmat = matfile['f']

print "d:"
print d
print "dmat:"
print dmat
 
diff = dmat - d
print "avg desc error:", (diff**2).sum(0).mean()
 
diff = fmat - f
print "avg frame error:", (diff**2).sum(0).mean()

