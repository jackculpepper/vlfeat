
run toolbox/vl_setup
 
im = imread('testfiles/000001.pgm');
[f,d] = vl_dsift(single(im), 'FloatDescriptors', 'Fast', ...
                 'Norm', 'Step', 2, 'Size', 4, 'WindowSize', 1.5);
  
save testfiles/000001_pgm_dsift_float.mat f d

[f,d] = vl_dsift(single(im), 'FloatDescriptors', 'Fast', ...
                 'Norm', 'Step', 2, 'Size', 4, 'WindowSize', 1.5, ...
                 'Color', 'RGB');
save testfiles/000001_pgm_dsift_float_color=rgb.mat f d


