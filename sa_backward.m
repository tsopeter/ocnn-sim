function z = sa_backward(E, a0)
    if (abs(E) < 0.0000001)
        z = 0.000001;
    else
        g = sa_forward(E, a0)/E;
        num = a0 * (E^2);
        dem = (1 + (E^2))^2;
        if abs(dem) == 0
            dem = 0.001;
        end
        a = exp(1+num/dem);
        z = g * a;
        if isnan(z)
            z = 0.001;
        end
        if isinf(z)
            z = 0.001;
        end
    end
end