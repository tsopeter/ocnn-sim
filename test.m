Nx = 2048;        %% number of elements
Ny = 2048;

nx = 100e-3;      %% dimensions of screen
ny = 100e-3;   

Kx = 500;         %% number of kernels
Ky = 500;

kx = Kx/nx;       %% size of each kernel
ky = Ky/ny;

% parameters
wavelength   = 632e-9;
radius       = 10 * nx;
focal_length = 80e-2;
distance     = 80e-2;
a0           = 20;

%c  = ones(dimx, dimy);

base = 1e-3; % assume a background radiation of 1 mW
P = 10e-3;  % assume a 10 mW source

kernel = ones(Nx, Ny); 
input  = wave_init(Nx, Ny, nx, ny, wavelength, box_mask(Nx, Ny, nx, ny, 0, 0, 10e-3, 40e-3, P));
result = forward_propagation(input, kernel, distance);

colormap('hot');
imagesc(abs(result.E));
title("Result");
colorbar;
