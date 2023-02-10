% define the parameters of the network

Nx = 2048;      % number of columns
Ny = 2048;      % number of rows

nx = 20e-3;
ny = 20e-3;

% interpolation value
ix = Nx/2;
iy = Ny/2;

P = 10e-3;  % 10 mW

wavelength = 1000e-9;

% create a plate to detect
plate = detector_plate(Nx, Ny, nx, ny, nx/4, nx/20);

% create a empty field
input = wave_init(Nx, Ny, nx, ny, wavelength, ones(Ny, Nx));

% we want to read the MNIST data
data  = read_MNIST('testing/images', 'testing/labels');

% get the interpolation value k
kx = log2(double(ix - data.n_cols)/double(data.n_cols - 1))+1;
ky = log2(double(iy - data.n_rows)/double(data.n_rows - 1))+1;

% get the lowest interpolation value
k = min(kx, ky);

% we want to initalize the kernel mask with random phase and amplitude
kernel = internal_random_amp(Nx, Ny);

% secondly, we want to resize he data such that the data has the same
% number of rows as the dimensions

% we want to observe it for 1028 by 1028 samples, or 50mm by 50mm

% to verify lets print out an image
v_img = data.images(1);

n = v_img.normalize();
n = interp2(n, k);
n = mask_resize(n, Nx, Ny);

% we can now multiply it, and propagate
input.E = input.E .* n * sqrt(P);

A1 = propagate(input.E, 10e-2, wavelength, Nx, Ny, nx, ny);

colormap('hot');
imagesc(abs(A1));
title("Result @ 10 cm");
colorbar;
figure;

A2 = propagate(input.E, 100e-2, wavelength, Nx, Ny, nx, ny);

colormap('hot');
imagesc(abs(A2));
title("Result @ 100 cm");
colorbar;


