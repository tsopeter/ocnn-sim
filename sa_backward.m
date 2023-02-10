function z = sa_backward(E, a0)
    g= 0.001;
    if abs(E) == 0
    else
        g = sa_forward(E, a0)/E;
    end
    num = a0 * (E^2);
    dem = (1 + (E^2))^2;
    if abs(dem) == 0
        dem = 0.001;
    end
    a = exp(1+num/dem);
    z = g * a;
end