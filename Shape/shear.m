clc;
clear all;
close all;

Im = imread('src_small_2.png');
h1 = figure; 
imshow(Im); 
title('Original Image');

a = -0.05;
T = maketform('affine', [1 0 0; a 1 0; 0 0 1] );
R = makeresampler({'cubic','nearest'},'fill');
B = imtransform(Im,T,R,'FillValues',[255 125 0]'); 
h2 = figure; imshow(B);
title('Sheared Image');