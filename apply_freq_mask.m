function output = apply_freq_mask(input, mask) % make sure that the mask and the input are the same dimensions
    f_input = fftshift(fft2(input));
    f_mask  = fftshift(fft2(mask)) ;
    output = ifft2(ifftshift(f_input .* f_mask));
end