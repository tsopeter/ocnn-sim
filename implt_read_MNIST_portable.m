function [images, labels, n_images, n_labels, n_rows, n_cols] = implt_read_MNIST_portable(datafile, labelfile)

    function [out, ss] = read_data (data, offset, n_bytes)
        out = 0.0;
        for i=offset:1:offset+n_bytes-1
            out = bitshift(out, 8) + data(i);
        end
        ss = offset+n_bytes;
    end

    function imgs = get_images(data, it)
        imgs = zeros(n_rows * n_cols, n_images);
        for q=1:1:n_images
            for j=1:1:(n_cols*n_cols)
                [z, it] = read_data(data, it, 1);
                imgs(j, q) = z;
            end
        end
    end

    function lbls = get_labels(data, it)
        lbls = zeros(n_labels, 1);
        for q=1:1:n_labels
            [z, it] = read_data(data, it, 1);
            lbls(q) = z;
        end
    end

    data_fd  = fopen(datafile);
    label_fd = fopen(labelfile);
    dfd      = fread(data_fd);
    lfd      = fread(label_fd);

    [magic_number, it] = read_data(dfd, 1, 4);

    if (magic_number ~= 2051)
        disp("error: data file has corruption or invalid file. magic number does not match.");
        return;
    end

    [n_images, it] = read_data(dfd, it, 4);
    [n_rows  , it] = read_data(dfd, it, 4);
    [n_cols  , it] = read_data(dfd, it, 4);

    images         = get_images(dfd, it);

    [magic_number, it] = read_data(lfd, 1, 4);

    if (magic_number ~= 2049)
        disp("error: label file has corruption or invalid file. magic number does not match.");
        return;
    end

    [n_labels, it] = read_data(lfd, it, 4);

    labels = get_labels(lfd, it);
end