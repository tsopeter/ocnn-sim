function output = internal_random_amp(Nx, Ny)
    output = zeros(Nx, Ny);
    for i=1:1:Nx
        for j=1:1:Ny
            output(i,j)=randn()+randn()*1j;
        end
    end
end