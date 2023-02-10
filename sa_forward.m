function z = sa_forward(E, a0)
    num = - a0/2;
    dem = 1 + E^2;
    z = exp(num/dem)*E;
end