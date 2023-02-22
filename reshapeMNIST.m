function Z = reshapeMNIST(X, Nx, Ny, ratio, k)
    function Q = internal_normalize(A)
        m = max(max(A));
        Q = A/m;
    end

    ix = round(Nx/ratio);
    iy = round(Ny/ratio);

    Z = zeros(Nx, Ny, length(X));


    for i=1:1:length(X)
        x = X(:,:,i);
        q = internal_normalize(x);
        r = interp2(q, k);
        R = mask_resize(r, Nx, Ny);
        Z(i) = R;
    end
end