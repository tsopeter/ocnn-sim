function z = sa_backward(E)
    a0 = 20;
    g = abs(E);
    p = angle(E);

    if (g < 0.0001)
        m = 0.0001;
        z = m * exp(1i * p);
    else
        q = abs(sa_forward(E))/g;
        m = q * exp(1+(a0*g*g)/(1+g^2)^2);
        z = m * exp(1i * p);
    end
end