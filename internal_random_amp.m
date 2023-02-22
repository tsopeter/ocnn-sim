function output = internal_random_amp(Nx, Ny)
    output = gpuArray(zeros(Nx, Ny, 'single'));
    for i=1:1:Nx
        for j=1:1:Ny
            v = randn();
            a = randn();
            output(i,j)=a * exp(1i * v);
        end
    end
end