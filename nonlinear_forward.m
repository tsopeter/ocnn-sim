function output = nonlinear_forward(input, a0)
    % for now, use SA_forward
    output = arrayfun(@(x)sa_forward(x), input);
end