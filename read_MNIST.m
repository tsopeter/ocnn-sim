function output = read_MNIST(datafile, labelfile)
    output = mnist;
    [images, labels, n_images, n_labels, n_rows, n_cols] = implt_read_MNIST(datafile, labelfile);

    % the images are placed as
    % images(1,:) = [a, b, c, ....]
    % images(2,:) = [a, b, c, ....]
    % ...
    % images(n_images,:) = [a, b, c, ....]ima

    % and the labels are too
    % labels(1,:) = [a]
    % labels(2,:) = [a]

    % we want to convert the image back into 28x28

    parsed_images(n_images) = v_image;
    for i=1:1:length(images)
        img  = v_image;
        vec = images(:,i);

        % we want to splice and concat
        temp = zeros(n_rows, n_cols);

        for j=1:1:n_cols
            stx = 1 + n_cols * (j - 1);
            ste = n_cols * j;
            temp(:,j)=vec(stx:ste, 1);
        end
        img.data = temp';
        img.n_rows = n_rows;
        img.n_cols = n_cols;
        parsed_images(i) = img;
    end

    output.images   = parsed_images;
    output.labels   = labels';
    output.n_images = n_images;
    output.n_labels = n_labels;
    output.n_rows   = n_rows;
    output.n_cols   = n_cols;
end