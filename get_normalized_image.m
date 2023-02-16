function nimg = get_normalized_image (img, Nx, Ny, k)
    nimg = mask_resize(interp2(img.normalize(), k), Nx, Ny);
end