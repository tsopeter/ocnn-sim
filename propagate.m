function output = propagate(input, distance, wavelength, Nx, Ny, nx, ny)
    dx = nx/Nx;     % size of each element
    dy = ny/Ny;
    
    rangex = 1/dx;  % number of frequencies available
    rangey = 1/dy;

    posx = linspace(-rangex/2, rangex/2, Nx);
    posy = linspace(-rangey/2, rangey/2, Ny);
    
    fftc = fftshift(fft2(input));
    
    [fxx, fyy] = meshgrid(posy, posx);
    
    kz = 2 * pi * sqrt((1/wavelength)^2 -(fxx.^2)-(fyy.^2));
    
    fftE = fftc .* exp(1j * (kz) * distance);
    output = ifft2(ifftshift(fftE));
end