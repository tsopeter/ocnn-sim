function output = circle_at(Nx, Ny, nx, ny, x, y, radius)
    output = zeros(Ny, Nx);

    dimx = nx/Nx;
    dimy = ny/Ny;

    for i=1:1:Ny
        for j=1:1:Nx
            xx = -nx/2 + (dimx * j) - x;
            yy = -ny/2 + (dimy * i) - y; 
            if (xx^2 + yy^2 <= radius^2)
                output(i, j) = 1;
            end
        end
    end
end