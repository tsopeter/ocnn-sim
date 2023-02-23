function nimg = get_normalized_image (img, Nx, Ny, k)
    nimg = mask_resize(gpuArray(interp2(img.normalize(), k)), Nx, Ny);
end