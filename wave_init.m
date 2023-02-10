function output = wave_init(Nx, Ny, nx, ny, wavelength, field)
    output            = wave;
    output.Nx         = Nx;
    output.Ny         = Ny;
    output.nx         = nx;
    output.ny         = ny;
    output.wavelength = wavelength;
    output.E          = gpuArray(field);
    output.distance   = 0;
end