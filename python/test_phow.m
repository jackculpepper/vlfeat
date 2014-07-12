
run toolbox/vl_setup

im = imread('testfiles/000001.pgm');
[f,d] = vl_phow(single(im), 'FloatDescriptors', 1, 'Verbose', 1, 'Step', 3);
save testfiles/000001_pgm_phow_float.mat f d

[f,d] = vl_phow(single(im), 'FloatDescriptors', 1, 'Verbose', 1, ...
                'Step', 3, 'Color', 'RGB');
save testfiles/000001_pgm_phow_float_color=rgb.mat f d
 
[f,d] = vl_phow(single(im), 'FloatDescriptors', 1, 'Verbose', 1, ...
                'Step', 3, 'Color', 'HSV');
save testfiles/000001_pgm_phow_float_color=hsv.mat f d


