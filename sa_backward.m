function z = sa_backward(E, a0)
    g = abs(E);
    p = angle(E);

    if (g < 0.0001)
        g = 0.0001;
    else
        g = abs(sa_forward(E, a0))/g;
        g = g * exp(1+(a0*g*g)/(1+g^2)^2);
    end
    z = g * exp(1i * p);
end