function dh = forward_propagation(batch, kernel, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, r1, r2, k, a0)
   img   = batch.img;
   label = batch.label;

   soln  = circle_at(Nx, Ny, nx, ny,r1, 0, r2);
   soln  = imrotate(soln, 36*label, 'crop');


   nimg  = mask_resize(interp2(img.normalize(), k), Nx, Ny);
   img_kernel = nimg .* kernel;
   img_prop_1 = propagate(img_kernel, distance_1, wavelength, Nx, Ny, nx, ny);
   img_non    = nonlinear_forward(img_prop_1, a0);
   img_prop_2 = propagate(img_non, distance_2, wavelength, Nx, Ny, nx, ny);

   img_det     = img_prop_2 .* plate;
   img_det_mag = abs(img_det).^2;

   dh = data_handler;
   dh.input_img  = nimg;
   dh.kernel_img = img_kernel;
   dh.distance_1_img = img_prop_1;
   dh.distance_2_img = img_prop_2;
   dh.nonlinear_img  = img_non;
   dh.result_img     = img_det;
   dh.given_label    = label;
   dh.soln_img       = soln;
   dh.result_label   = detector_location(img_det_mag, Nx, Ny, nx, ny, r1, r2);
end