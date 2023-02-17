function zh = backward_propagation(dh, d1, d2, a0)
    i0 = dh.input_img;
    i2 = dh.distance_1_img;
    D  = dh.distance_2_img;
    S  = dh.soln_img;

    % remember
    % i0 <- input
    % i1 = i0 .* kernel
    % i2 = afm(i1, d1)
    % i3 = sigma(i2)
    % D  = afm(i3, d2)
    % S  = solution

    % we need to take the derivative
    % with respect to input i3
    dD      = D - S;    % from the quadratic cost function

    % dD_di3 = afm(180(d2), dD)
    dD_di3  = apply_freq_mask(flip_180(d2), dD);

    % take derivative of i3 with respect to i2
    di3_di2 = nonlinear_backward(i2, a0);

    % take derivative of i2 with respect to i1
    di3_di1 = apply_freq_mask(flip_180(d1), di3_di2);

    % take derivative of i1 with respect to dk
    di1_dk  = i0;

    % compute backpropagation to change the kernel
    % all other parameters such as distance_1 and distance_2
    % are to be fixed.
    dD_dk   = dD_di3 .* di3_di1 .* di1_dk;
    zh = data_handler;
    zh.nabla = dD_dk;

end