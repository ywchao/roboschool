im = imread('checker_blue.png');
im = im(:,:,[3 2 1]);
imwrite(im, 'checker_blue_roboschool.png');
