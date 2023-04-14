function output = internal_random_amp(Nx, Ny)
    output = gpuArray(zeros(Nx, Ny, 'single'));
    for x=1:1:Nx
        for y=1:1:Ny
            z = exp(-1i * randn());
            output(x,y)=z;
        end
    end
end