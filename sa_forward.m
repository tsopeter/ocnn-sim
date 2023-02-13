function z = sa_forward(E, a0)
    % preserve phase
    g = abs(E);
    p = angle(E);

    g = exp(-a0/2/(1+g^2));
    z = g * exp(1i*p);
end