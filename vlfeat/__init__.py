import _vlfeat
import numpy
from quickshift import vl_quickseg,vl_quickvis

def vl_sift(
		data, 
		frames=numpy.zeros(1), 
		octaves=-1, 
		levels=-1, 
		first_octave=0,
		peak_thresh=-1.0, 
		edge_thresh=-1.0,
		norm_thresh=-1.0,
		magnif=-1.0,
		window_size=-1.0,
		orientations=False,
		verbose=0):
	""" Computes the SIFT frames [1] (keypoints) F of the image I. I is a 
	gray-scale image in single precision. Each column of F is a feature frame 
	and has the format [X;Y;S;TH], where X,Y is the (fractional) center of the 
	frame, S is the scale and TH is the orientation (in radians). 
	Computes the SIFT descriptors [1] as well. Each column of D is the 
	descriptor of the corresponding frame in F. A descriptor is a 
	128-dimensional vector of class UINT8. 
	
	@param data         A gray-scale image in single precision 
	                    (float numpy array).
	@param frames       Set the frames to use (bypass the detector). If frames 
	                    are not passed in order of increasing scale, they are 
	                    re-orderded. 
	@param octaves      Set the number of octave of the DoG scale space. 
	@param levels       Set the number of levels per octave of the DoG scale 
	                    space. The default value is 3.
	@param first_octave Set the index of the first octave of the DoG scale 
	                    space. The default value is 0.
	@param peak_thresh  Set the peak selection threshold. 
	                    The default value is 0. 
	@param edge_thresh  Set the non-edge selection threshold. 
	                    The default value is 10.
	@param norm_thresh  Set the minimum l2-norm of the descriptor before 
	                    normalization. Descriptors below the threshold are set 
	                    to zero.
	@param magnif       Set the descriptor magnification factor. The scale of 
	                    the keypoint is multiplied by this factor to obtain the
	                    width (in pixels) of the spatial bins. For instance, if
	                    there are there are 4 spatial bins along each spatial
	                    direction, the ``diameter'' of the descriptor is
	                    approximatively 4 * MAGNIF. The default value is 3.
	@param orientations Compute the orientantions of the frames overriding the 
	                    orientation specified by the 'Frames' option.
	@param verbose      Be verbose (may be repeated to increase the verbosity
	                    level). 
	"""
	if not data.flags['F_CONTIGUOUS']:
		data = numpy.array(data, order='F')		
	if not frames.flags['F_CONTIGUOUS']:
		frames = numpy.array(frames, order='F')
		
	return _vlfeat.vl_sift(data, frames, octaves, levels, first_octave, 
						peak_thresh, edge_thresh, norm_thresh, magnif,
						window_size, orientations, verbose)

def vl_mser(
		data, 
		delta=5, 
		max_area=.75, 
		min_area=.0002,
		max_variation=.25, 
		min_diversity=.2):
	""" Computes the Maximally Stable Extremal Regions (MSER) [1] of image I 
	with stability threshold DELTA. I is any array of class UINT8. R is a vector
	of region seeds. \n 
	A (maximally stable) extremal region is just a connected component of one of
	the level sets of the image I. An extremal region can be recovered from a
	seed X as the connected component of the level set {Y: I(Y) <= I(X)} which
	contains the pixel o index X. \n
	It also returns ellipsoids F fitted to the regions. Each column of F 
	describes an ellipsoid; F(1:D,i) is the center of the elliposid and
	F(D:end,i) are the independent elements of the co-variance matrix of the
	ellipsoid. \n
	Ellipsoids are computed according to the same reference frame of I seen as 
	a matrix. This means that the first coordinate spans the first dimension of
	I. \n
	The function vl_plotframe() is used to plot the ellipses.
	
	@param data           A gray-scale image in single precision.
	@param delta          Set the DELTA parameter of the VL_MSER algorithm. 
	                      Roughly speaking, the stability of a region is the
	                      relative variation of the region area when the
	                      intensity is changed of +/- Delta/2. 
	@param max_area       Set the maximum area (volume) of the regions relative 
	                      to the image domain area (volume). 
	@param min_area       Set the minimum area (volume) of the regions relative 
	                      to the image domain area (volume). 
	@param max_variation  Set the maximum variation (absolute stability score) 
	                      of the regions. 
	@param min_diversity  Set the minimum diversity of the region. When the 
	                      relative area variation of two nested regions is below 
	                      this threshold, then only the most stable one is 
	                      selected. 
	"""
	if not data.flags['F_CONTIGUOUS']:
		data = numpy.array(data, order='F')		
		
	return _vlfeat.vl_mser(data, delta, max_area, min_area, \
							max_variation, min_diversity)
	

def vl_erfill(data, r):
	""" Returns the list MEMBERS of the pixels which belongs to the extremal
	region represented by the pixel ER. \n
	The selected region is the one that contains pixel ER and of intensity 
	I(ER). \n
	I must be of class UINT8 and ER must be a (scalar) index of the region
	representative point. 
	"""
	if not data.flags['F_CONTIGUOUS']:
		data = numpy.array(data, order='F')
		
	return _vlfeat.vl_erfill(data, r)


def vl_dsift(
			data, 
			step=-1, 
			bounds=numpy.zeros(1, 'f'), 
			size=-1, 
			fast=True, 
			verbose=False, 
			norm=False):
	""" [F,D] = VL_DSIFT(I) calculates the Dense Histogram of Gradients (DSIFT) 
	descriptors for the image I. I must be grayscale in SINGLE format.\n\n
	
	In this implementation, a DSIFT descriptor is equivalent to a SIFT 
	descriptor (see VL_SIFT()). This function calculates quickly a large number
	of such descriptors, for a dense covering of the image with features of the
	same size and orientation.\n\n
	
	The function returns the frames F and the descriptors D. Since all frames
	have identical size and orientation, F has only two rows (for the X and Y
	center coordinates). The orientation is fixed to zero. The scale is related
	to the SIZE of the spatial bins, which by default is equal to 3 pixels (see
	below). If NS is the number of bins in each spatial direction (by default
	4), then a DSIFT keypoint covers a square patch of NS by SIZE pixels.\n\n
	
	@remark The size of a SIFT bin is equal to the magnification factor MAGNIF 
	(usually 3) by the scale of the SIFT keypoint. This means that the scale of
	the SIFT keypoints corresponding to the DSIFT descriptors is SIZE / MAGNIF. 
	
	@remark Although related, DSIFT is not the same as the HOG descriptor used 
	in [1]. This descriptor is equivalent to SIFT instead. 
		
	@param step    Extract a descriptor each STEP pixels.
	@param size    A spatial bin covers SIZE pixels.
	@param norm    Append the frames with the normalization factor applied to 
	               each descriptor. In this case, F has 3 rows and this value 
	               is the 3rd row. This information can be used to suppress
	               descriptors with low contrast.
	@param fast    Use a flat rather than Gaussian window. Much faster.
	@param verbose Be verbose. 
	"""
	if not data.flags['F_CONTIGUOUS']:
		data = numpy.array(data, order='F')		
		
	return _vlfeat.vl_dsift(data, step, bounds, size, fast, verbose, norm)


def vl_siftdescriptor(grad, frames):
	""" D = VL_SIFTDESCRIPTOR(GRAD, F) calculates the SIFT descriptors of the 
	keypoints F on the pre-processed image GRAD. GRAD is a 2xMxN array. The 
	first layer GRAD(1,:,:) contains the modulus of gradient of the original 
	image modulus. The second layer GRAD(2,:,:) contains the gradient angle 
	(measured in radians, clockwise, starting from the X axis -- this assumes 
	that the Y axis points down). The matrix F contains one column per keypoint 
	with the X, Y, SGIMA and ANLGE parameters. \n \n

	In order to match the standard SIFT descriptor, the gradient GRAD should be 
	calculated after mapping the image to the keypoint scale. This is obtained 
	by smoothing the image by a a Gaussian kernel of variance equal to the scale 
	of the keypoint. Additionaly, SIFT assumes that the input image is 
	pre-smoothed at scale 0.5 (this roughly compensates for the effect of the 
	CCD integrators), so the amount of smoothing that needs to be applied is 
	slightly less. The following code computes a standard SIFT descriptor by 
	using VL_SIFTDESCRIPTOR(): 
	"""
	if not grad.flags['F_CONTIGUOUS']:
		grad = numpy.array(grad, order='F')
	if not frames.flags['F_CONTIGUOUS']:
		frames = numpy.array(frames, order='F')
		
	return _vlfeat.vl_siftdescriptor(grad, frames)

def vl_imsmooth(I, sigma):		
	""" I=VL_IMSMOOTH(I,SIGMA) convolves the image I by an isotropic Gaussian 
	kernel of standard deviation SIGMA. I must be an array of doubles. IF the 
	array is three dimensional, the third dimension is assumed to span different
	channels (e.g. R,G,B). In this case, each channel is convolved 
	independently.
	"""
	if not I.flags['F_CONTIGUOUS']:
		I = numpy.array(I, order='F')
	return _vlfeat.vl_imsmooth(I, sigma)


def vl_ikmeans(data, K, max_niters=200, method='lloyd', verbose=0):
	""" Integer K-means.
	[C, I] = VL_IKMEANS(X,K) returns the centers of a K-means partitioning of
	the data space X the cluster associations I of the data. X must be of class 
	UINT8. C is of class UINT32.\n\n
	
	VL_IKMEANS() accepts the following options: \n
	
	@param max_niters  Maximum number of iterations before giving up (the 
	                   algorithm stops as soon as there is no change in the data
	                   to cluster associations).
	@param method      Algorithm to use ('Lloyd', 'Elkan').
	@param verbose     Increase the verbosity level.
	"""
	if not data.flags['F_CONTIGUOUS']:
		data = numpy.array(data, order='F')
	return _vlfeat.vl_ikmeans(data, K, max_niters, method, verbose)

def vl_ikmeanspush(data, centers, method='lloyd', verbose=0):
	""" VL_IKMEANSPUSH  Project data on integer K-means partitions
	I = VL_IKMEANSPUSH(X,C) projects the data X to the integer K-means clusters
	of centers C returning the cluster indices I.
	"""
	if not data.flags['F_CONTIGUOUS']:
		data = numpy.array(data, order='F')
	if not centers.flags['F_CONTIGUOUS']:
		centers = numpy.array(centers, order='F')
	return _vlfeat.vl_ikmeanspush(data, centers, method, verbose)
	
def vl_binsum(H, X, B, DIM=-1):
	"""
	"""	
	if not H.flags['F_CONTIGUOUS']:
		H = numpy.array(H, order='F')
	if not X.flags['F_CONTIGUOUS']:
		X = numpy.array(X, order='F')
	if not B.flags['F_CONTIGUOUS']:
		B = numpy.array(B, order='F')
	return  _vlfeat.vl_binsum(H, X, B, DIM)

def vl_hikmeans(data, K, nleaves, verb=0, max_iters=200, method='lloyd'):
	"""
	"""
	if not data.flags['F_CONTIGUOUS']:
		data = numpy.array(data, order='F')
	return _vlfeat.vl_hikmeans(data, K, nleaves, verb, max_iters, method)
		
def vl_hikmeanspush(tree, data, verb=0, method='lloyd'):
	"""
	"""
	if not data.flags['F_CONTIGUOUS']:
		data = numpy.array(data, order='F')
	return _vlfeat.vl_hikmeanspush(tree, data, verb, method)
		

def vl_rgb2gray(data):
	""" Rgb 2 gray consersion giving the same result as matlab own conversion 
	function.
	@param data A color image as 3D numpy array.
	@return A gray image as 2D numpy array (type is float but numbers are 
	rounded)	
	"""
	return numpy.round(0.2989 * data[:,:,0] + 0.5870 * data[:,:,1] + 0.1140 * data[:,:,2])



def vl_phow(im, norm=True, fast=True, step=3, sizes=[4, 6, 8, 10],
            window_size=1.5, magnif=6.0, verbose=True,
            contrast_threshold=0.005):
    import numpy as np

    descrs = np.zeros((128,0), dtype=np.single)
    frames = np.zeros((4,0), dtype=np.double)

    for si in range(len(sizes)):

        # Recall from VL_DSIFT() that the first descriptor for scale SIZE has
        # center located at XC = XMIN + 3/2 SIZE (the Y coordinate is
        # similar). It is convenient to align the descriptors at different
        # scales so that they have the same geometric centers. For the
        # maximum size we pick XMIN = 1 and we get centers starting from
        # XC = 1 + 3/2 MAX(OPTS.SIZES). For any other scale we pick XMIN so
        # that XMIN + 3/2 SIZE = 1 + 3/2 MAX(OPTS.SIZES).
        #
        # In practice, the offset must be integer ('bounds'), so the
        # alignment works properly only if all OPTS.SZES are even or odd.

        off = np.floor(3.0/2.0 * (max(sizes) - sizes[si]))
        bounds = np.array([off, off, np.infty, np.infty], dtype=np.double)

        # scale space
        sigma = sizes[si] / magnif
        ims = vl_imsmooth(im, sigma)

        ## setting norm=True means f will have 3 dims
        f, d = vl_dsift(ims, norm=True, fast=True, step=step, bounds=bounds,
                        size=sizes[si], window_size=window_size,
                        verbose=verbose)

        # zero out low contrast descriptors
        idx = f[2,:] < contrast_threshold
        d[:,idx] = 0

        # append the scale so we have x,y,norm,scale
        scale = np.ones((1,f.shape[1]), dtype=np.double) * sizes[si]
        f = np.vstack([f, scale])


        descrs = np.hstack([descrs, d])
        frames = np.hstack([frames, f])

    ## use 1-based pixel numbering, as the matlab impl does
    frames[:2,:] += 1

    return frames, descrs


def vl_phow_color(im, norm=True, fast=True, step=3, sizes=[4, 6, 8, 10],
                  window_size=1.5, magnif=6.0, verbose=True,
                  contrast_threshold=0.005, color='rgb', debug=False):
    import numpy as np

    if debug:
        import scipy.io

    descrs = np.zeros((384,0), dtype=np.single)
    frames = np.zeros((4,0), dtype=np.double)

    # make sure we get an image with 3 color channels to start with
    assert len(im.shape) == 3

    channels = im.shape[2]

    # standardize the image
    if color == 'rgb':
        pass
    elif color == 'opponent':
        mu = 0.3*im[:,:,0] + 0.59*im[:,:,1] + 0.11*im[:,:,2]
        alpha = 0.01

        im_opponent = im.copy()
        im_opponent[:,:,0] = mu
        im_opponent[:,:,1] = (im[:,:,0] - im[:,:,1])/np.sqrt(2) + alpha*mu
        im_opponent[:,:,2] = \
            (im[:,:,0] + im[:,:,1] - 2*im[:,:,2])/np.sqrt(6) + alpha*mu

        im = im_opponent

    for si in range(len(sizes)):

        # Recall from VL_DSIFT() that the first descriptor for scale SIZE has
        # center located at XC = XMIN + 3/2 SIZE (the Y coordinate is
        # similar). It is convenient to align the descriptors at different
        # scales so that they have the same geometric centers. For the
        # maximum size we pick XMIN = 1 and we get centers starting from
        # XC = 1 + 3/2 MAX(OPTS.SIZES). For any other scale we pick XMIN so
        # that XMIN + 3/2 SIZE = 1 + 3/2 MAX(OPTS.SIZES).
        #
        # In practice, the offset must be integer ('bounds'), so the
        # alignment works properly only if all OPTS.SZES are even or odd.

        off = np.floor(3.0/2.0 * (max(sizes) - sizes[si]))
        bounds = np.array([off, off, np.infty, np.infty], dtype=np.double)

        # scale space
        sigma = sizes[si] / magnif
        ims = vl_imsmooth(im, sigma)

        d_color = []
        f_color = []
        for ch in range(channels):
            ## setting norm=True means f will have 3 dims
            f, d = vl_dsift(ims[:,:,ch], norm=True, fast=True,
                            step=step, bounds=bounds,
                            size=sizes[si], window_size=window_size,
                            verbose=verbose)

            #print f[:,:10].T

            ## use 1-based pixel numbering, as the matlab impl does
            f[:2,:] += 1


            if debug:
                matfile = scipy.io.loadmat('testfiles/000001_jpg_phow_float_color=rgb_si=%d_ch=%d.mat' % (si+1, ch+1))

                dmat = matfile['dk']
                fmat = matfile['fk']

                print "d:"
                print d
                print "dmat:"
                print dmat

                diff = dmat - d
                print "avg desc error:", (diff**2).sum(0).mean()

                diff = fmat - f
                print "avg frame error:", (diff**2).sum(0).mean()


            d_color.append(d)
            f_color.append(f)


        # concat descriptors from color channels together to get 384xN
        d = d_color[0]
        for ch in range(channels-1):
            d = np.vstack([d, d_color[ch+1]])

        # compute contrast
        if color == 'rgb':
            contrast = np.mean([ f_color[0][2,:],
                                 f_color[1][2,:],
                                 f_color[2][2,:] ], axis=0)
            #contrast = f_color[0][2,:]
        elif color == 'opponent':
            contrast = f_color[0][2,:]


        if debug:
            matfile = scipy.io.loadmat('testfiles/000001_jpg_phow_float_color=rgb_si=%d_contrast.mat' % (si+1))
            contrastmat = matfile['contrast']
            diff = contrastmat - contrast
            print "avg contrast error:", (diff**2).sum(0).mean()

        # zero out low contrast descriptors
        idx = contrast < contrast_threshold
        d[:,idx] = 0

        # use the frame from the first color channel, to match matlab code
        f = f_color[0]

        # append the scale so we have x,y,norm,scale
        scale = np.ones((1,f.shape[1]), dtype=np.double) * sizes[si]
        f = np.vstack([f, scale])


        descrs = np.hstack([descrs, d])
        frames = np.hstack([frames, f])

    return frames, descrs



