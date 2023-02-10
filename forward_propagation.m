function output = forward_propagation(input, kernels, distance)
    % we take the hprod of input and kernel mask, then propagate a distance

    E1 = input.E .* kernels;
    E2 = propagate(E1, distance, input.wavelength, input.Nx, input.Ny, input.nx, input.ny);
    
    output = wave;
    output.wavelength = input.wavelength;
    output.Nx         = input.Nx;
    output.Ny         = input.Ny;
    output.nx         = input.nx;
    output.ny         = input.ny;
    output.E          = E2;
    output.distance   = input.distance + distance;
end