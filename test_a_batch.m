function z = test_a_batch(test, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, a0, M)
    z = 0;
    parfor (i=1:length(test), M)
        batch = test(i);
        handle = forward_propagation(batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, a0);
        handle.result_label = detector_location(handle.result_img, Nx, Ny, nx, ny, r1, r2);
        if (handle.result_label == handle.given_label)
            z = z + 1;
        end
    end
end