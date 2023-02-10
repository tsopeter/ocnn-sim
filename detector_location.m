function location = detector_location(input, Nx, Ny, nx, ny, radius1, radius2)

    % at every 36 degrees, put a circle of radius2 at radius1

    temp = abs(circle_at(Nx, Ny, nx, ny, radius1, 0, radius2)).^2;
    m = sum(sum(temp));
    j = 0;
    for r=1:1:9
        plate  = circle_at(Nx, Ny, nx, ny, radius1, 0, radius2);
        plate  = imrotate(plate, 36*r, 'crop');

        result = input .* plate;
        z      = sum(sum(abs(result).^2));
        if (z > m)
            j = r;
            m = z;
        end
    end
    location = j;
end