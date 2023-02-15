function z = test_a_batch(test, kernel, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, r1, r2, k, a0, M)
    z = 0;
    parfor (i=1:length(test), M)
        batch = test(i);
        handle = forward_propagation(batch, kernel, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, r1, r2, k, a0);
        if (handle.result_label == handle.given_label)
            z = z + 1;
        end
    end
end