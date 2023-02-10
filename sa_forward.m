function z = sa_forward(E, a0)
    num = - a0/2;
    dem = 1 + E^2;
    if abs(dem) == 0
        dem = 0.001;
    end
    z = exp(num/dem)*E;
end