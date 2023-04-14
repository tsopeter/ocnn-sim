function z = sa_backward(E)
    a0 = 20;
    g = abs(E);
    p = angle(E);

    m = exp(-a0/2/(1+g^2));
    q = m * (1+(a0*g*g)/(1+g^2)^2);
    z = q * exp(1i * p);
end