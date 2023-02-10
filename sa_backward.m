function z = sa_backward(E, a0)
    g = sa_forward(E, a0)/E;
    num = a0 * (E^2);
    dem = (1 + (E^2))^2;
    a = exp(1+num/dem);
    z = g * a;
end