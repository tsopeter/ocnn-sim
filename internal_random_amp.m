function output = internal_random_amp(Nx, Ny)
    output = gpuArray(zeros(Nx, Ny, 'single'));
    for i=1:1:Nx
        for j=1:1:Ny
            z = randn()+randn()*1i;
            %g = abs(z);
            output(i,j)=z;
        end
    end
end