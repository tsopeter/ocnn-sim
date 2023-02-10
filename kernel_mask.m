function output = kernel_mask(input, kernel, kx, ky, i, j)
    % we need to map the ith and jth kernel to the ith and jth pixel
    ithe = i * kx;
    jthe = j * ky;

    iths = ithe - kx + 1;
    jths = jthe - ky + 1;

    input(iths:ithe, jths:jthe)=kernel;     % this is how we address the kernels
    output = input;
end