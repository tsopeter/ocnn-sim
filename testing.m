function z = testing(test, plate, kernel, d1, d2, Nx, Ny, k, a0, n_bars, ix, iy, M)
    z = 0;
    parfor (i=1:length(test), M)
        batch = test(i);
        handle = forward_propagation(batch, plate, kernel, d1, d2, Nx, Ny, k, a0, n_bars);
        handle.result_label = detector_location(handle.result_img, Nx, Ny, ix, iy, n_bars);
        if (handle.result_label == handle.given_label)
            z = z + 1;
        end
    end
end