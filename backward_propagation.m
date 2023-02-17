function zh = backward_propagation(dh, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, a0)
    i0 = dh.input_img;
    i2 = dh.distance_1_img;
    %D3 = dh.distance_2_img;
    D  = dh.result_img;
    S  = dh.soln_img;

    m  = plate;

    %
    %   Let the following structure define the network
    %   K  -> D1 -> N  -> D2 -> M  -> D
    %   R1 -> R2 -> R3 -> R4 -> R5
    %   
        
    d1      = get_propagation_distance(Nx, Ny, nx, ny, distance_1, wavelength);
    d2      = get_propagation_distance(Nx, Ny, nx, ny, distance_2, wavelength);

    dD      = D - S;
    dD_di4  = m;
    dD_di3  = apply_freq_mask(flip_180(d2), dD_di4);
    di3_di2 = nonlinear_backward(i2, a0);
    di3_di1 = apply_freq_mask(flip_180(d1), di3_di2);
    di1_dk  = i0;

    % compute backpropagation to change the kernel
    % all other parameters such as distance_1 and distance_2
    % are to be fixed.
    dD_dk   = dD .* dD_di3 .* di3_di1 .* di1_dk;
    zh = data_handler;
    zh.nabla = dD_dk;

end