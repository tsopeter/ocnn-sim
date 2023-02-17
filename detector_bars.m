function z = detector_bars(Nx, Ny, label, n_bars)
    n = zeros(Ny, Nx);

    n_width = floor(Ny/n_bars);

    for i=1+(label*n_width):n_width*10:Ny
        n(i:i+n_width-1,:)=ones(n_width, Nx);
        n(i+n_width-1, :) = zeros(1, Nx);
    end

    z = n(1:Ny, :);
end