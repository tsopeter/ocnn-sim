function dh = forward_propagation(batch, plate, kernel, d1, d2, Nx, Ny, k, a0, n_bars)
   img   = batch.img;
   label = batch.label;

   soln  = detector_bars(Nx, Ny, label, n_bars);

   nimg  = get_normalized_image(img, Nx, Ny, k);
   img_kernel = nimg .* kernel;
   img_prop_1 = apply_freq_mask(img_kernel, d1);
   img_non    = nonlinear_forward(img_prop_1, a0);
   img_prop_2 = apply_freq_mask(img_non, d2);

   img_det     = img_prop_2 .* plate;

   dh = data_handler;
   dh.input_img  = nimg;
   dh.distance_1_img = img_prop_1;
   dh.distance_2_img = img_prop_2;
   dh.result_img     = img_det;
   dh.soln_img       = soln;
   dh.given_label    = label;
end