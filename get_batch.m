function batch = get_batch(data, n_perbatch, random_flag)
    n_images = data.n_images;

    if (random_flag==1)
        r = randi([1, n_images], 1, n_perbatch);
    else
        r = 1:n_perbatch;
    end

    batch(n_perbatch) = v_batch;
    for i=1:1:n_perbatch
        temp = v_batch;
        temp.img   = data.images(r(i));
        temp.label = data.labels(r(i));
        batch(i) = temp;
    end

end
