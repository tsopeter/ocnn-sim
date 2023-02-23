function output = nonlinear_backward(input, a0)
    % for now, use SA_backwaard
    output = arrayfun(@(x)sa_backward(x), input);
end