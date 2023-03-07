function dLdX = implt_derivation_softmax(dLdZ, X, Z)
    W = size(dLdZ);
    %
    %
    % Create the Jacobian, which is an N by N matrix
    J = gpuArray(zeros(W(1), W(1), 'zeros'));

    for i=W(1)
        for j=W(1)
            if i==j
                J(i,j)=Z(i) * (1 - Z(j));
            else
                J(i,j)=-1 * Z(i) * Z(j);
            end
        end
    end


    % The Jacobian is arranged as such
    %
    %   | dZ1dX1    dZ1dX2  ... dZ1dXN  |
    %   | dZ2dX1    dZ2dX2  ... dZ2dXN  |
    %   | ...                           |
    %   | dZNdX1    dZNdX2  ... dZNdXN  |
    % 
    %   Therefore, derivative dLdX = [dLdX1, dLdX2, ..., dLdXN]
    %   where dLdXi = dLdZ1 * dZ1dXi + dLdZ2 * dZ2dXi + ... + dLdZN * dZNdXi
    %   
    %   Let dLdX be the row-major vector above
    %   Ket dLdZ be [dLdZ1, dLdZ2, ..., dLdZN] be a row-major vector.
    %   Then dLdXi is simply the inner product of the two vectors, or
    %   
    %   dLdXi = dLdZ .* J(:,i);

    dLdX = gpuArray(zeros(W, 'single'));

    for i=W(1)
        dLdX(i) = dLdZ .* J(:, i);
    end

end