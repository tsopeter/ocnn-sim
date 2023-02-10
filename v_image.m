classdef v_image
    properties
        data;
        n_rows;
        n_cols;
    end

    methods
        function out = normalize(obj)
            if (length(obj.data) <= 0)
                out = [];
            else 
                m = max(max(abs(obj.data)));
                if (m == 0)
                    out = obj.data;
                else
                    out = obj.data / m;
                end
            end
        end
    end
end