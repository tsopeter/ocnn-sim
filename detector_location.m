function z = detector_location(input, Nx, Ny, ix, iy, n_bars)

    % at every 36 degrees, put a circle of radius2 at radius1
    z = 0;
    m_max    = 0;

    for i=0:1:9
        x = mask_resize(detector_bars(ix, iy, i, n_bars), Nx, Ny);
        r = input .* x;
        s = sum(sum(abs(r).^2));
        if (s > m_max)
            m_max = s;
            z = i;
        end
    end
end