function z = test_a_batch(test, kernel, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, r1, r2, k, a0)
    z = 0;
    for batch=test.batch
        handle = forward_propagation(batch, kernel, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, r1, r2, k, a0);
        handle.result_label = detector_location(handle.result_img, Nx, Ny, nx, ny, r1, r2);
        if (handle.result_label == handle.given_label)
            z = z + 1;
        end
    end
end