function dh = forward_propagation(batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, a0)
   img   = batch.img;
   label = batch.label;

   soln  = imrotate(circle_at(Nx, Ny, nx, ny,r1, 0, r2), 36*label, 'crop');

   nimg  = get_normalized_image(img, Nx, Ny, k);
   img_kernel = nimg .* kernel;
   img_prop_1 = conv2(img_kernel, d1, 'same');
   img_non    = nonlinear_forward(img_prop_1, a0);
   img_prop_2 = conv2(img_non, d2, 'same');

   img_det     = img_prop_2 .* plate;

   dh = data_handler;
   dh.input_img  = nimg;
   dh.distance_1_img = img_prop_1;
   dh.distance_2_img = img_prop_2;
   dh.result_img     = img_det;
   dh.soln_img       = soln;
   dh.given_label    = label;
end