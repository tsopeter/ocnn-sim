function zh = backward_propagation(dh, Nx, Ny, nx, ny, r1, r2, rd1, rd2, a0, P)
    i0 = dh.input_img;
    i2 = dh.distance_1_img;
    D  = dh.distance_2_img;
    
    %
    %
    %
    % continue pass of forward propagation
    W = size(D);
    results   = gpuArray(zeros(W(1), W(2), 10, 'single'));
    results1D = gpuArray(zeros(10, 1, 'single'));
    expected  = gpuArray(zeros(10, 1, 'single'));
    expected(dh.given_label+1)=1;

    for i=1:10
        plate = imrotate(circle_at(Nx, Ny, nx, ny, r1, 0, r2), 36*(i-1), 'crop');
        results(:,:,i)=abs(D) .* plate;
        results1D(i) = sum(sum(results(:,:,i)));
    end

    %
    %
    %
    % pass through softmax layer
    %
    sfmax = softmax(results1D);

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
    dD      = abs(sfmax) - expected;    % from the quadratic cost function
    
    %
    %
    % backward propagate through softmax layer
    dDdsfmax = implt_derivation_softmax(dD, results1D, sfmax);

    %
    %
    %
    % now, backward propagate, flatten layer
    %
    % Remember that, Zi = sum(sum(X .* Pi));
    % where i denotes a label
    %
    %
    % dDdsfmax = [dLDZ0, dLdZ1, ..., dLdZN]
    %
    %
    % then, dLdX = dLdZ0 * dZ0dX + dLdZ1 * dZ1dX ... + dLdZN * dZNdX
    %
    %
    % What is dZidX ??
    %
    % Zi = sum(sum(X .* Pi));
    %
    % dZidX = Pi
    %

    size_D = size(D);
    dDmask = gpuArray(zeros(size_D, 'single'));
    for i=1:10
        Pmi = imrotate(circle_at(Nx, Ny, nx, ny, r1, 0, r2), 36*(i-1), 'crop');
        dDmask = dDmask + dDdsfmax(i) * Pmi;
    end

    % dD_di3 = afm(180(d2), dD)

    function Z = backprop_internal(dDf)
        dD_di3 = conv2(rd2, dDf, 'full');
        dD_di2 = dD_di3 .* conj(nonlinear_backward(i2, a0));
        dD_di1 = conv2(rd1, dD_di2, 'full');
        Z = i0 .* dD_di1;
    end

    zh = data_handler;
    zh.nabla = backprop_internal(dDmask);
end