function output = internal_random_amp(Nx, Ny)
    output = gpuArray(zeros(Nx, Ny, 'single'));
    for i=1:1:Nx
        for j=1:1:Ny
            v = normrnd(0, 1);
            a = normrnd(0, 1);
            g = a + 1i * v;
            output(i,j)= g / abs(g);
        end
    end
end