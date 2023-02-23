function z = kernelLimit(k)
    function q = limit(x)
        if x > 255
            q = 255 .* angle(x);
        else
            q = x;
        end
    end

    % imposes a 255 limit
    z = arrayfun(@(x)limit(abs(x)), k);
end