function zh = backward_propagation(t, dh, rd1, rd2, a0, P)
    i0 = dh.input_img;
    i2 = dh.distance_1_img;
    D  = dh.distance_2_img;
    S  = dh.soln_img .* sqrt(P);

    % remember
    % i0 <- input
    % i1 = i0 .* kernel
    % i2 = afm(i1, d1)
    % i3 = sigma(i2)
    % D  = afm(i3, d2)
    % S  = solution
    %


    % we need to take the derivative
    % with respect to input i3
    dD      = abs(D) - S;   % from the quadratic cost function

    dD_di3  = conv2(rd2, dD, 'full');

    % take derivative of i3 with respect to i2
    dD_di2 = dD_di3 .* conj(nonlinear_backward(i2, a0));

    % take derivative of i2 with respect to i1
    dD_di1 = conv2(rd1, dD_di2, 'full');

    % take derivative of i1 with respect to dk
    dD_dk  = i0 .* dD_di1;

    zh = data_handler;
    zh.nabla = dD_dk;

end