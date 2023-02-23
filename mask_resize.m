function output = mask_resize(input, Nx, Ny)    % function resizes and places in the middle. Assumes that the input is smaller than the result
    output = zeros(Ny, Nx, 'single');

    % get the center of the matrix
    centerx = round(Nx/2);
    centery = round(Ny/2);

    % get the dimensions of input matrix
    inputx = round(length(input)/2);
    inputy = round(length(input(1,:))/2);


    startx = centerx - inputx;
    starty = centery - inputy;

    endx   = centerx + inputx;
    endx   = endx - ((endx - startx) - length(input)) - 1;
    endy   = centery + inputy;
    endy   = endy - ((endy - starty) - length(input(1,:))) - 1;

    output(starty:endy, startx:endx) = input;
end