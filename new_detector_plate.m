function z = new_detector_plate(Nx, Ny, n_bars)
    z = zeros(Ny, Nx);

    for i=0:9
        z = z + detector_bars(Nx, Ny, i, n_bars);
    end
end