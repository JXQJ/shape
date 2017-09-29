Im = imread('roi_200_tilt.png');
Im = rgb2gray(Im);
BW = imbinarize(Im);

figure
imshowpair(Im, BW,'montage')
imwrite(BW, 'roi_200_tilted.png')