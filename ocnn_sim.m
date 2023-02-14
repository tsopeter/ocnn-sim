% optical artifical neural network
% 
%

clear;

% define the parameters of the network

        Nx = 512;      % number of columns
        Ny = 512;      % number of rows
        
        % this defines the size of the display
        nx = 20e-2;
        ny = 20e-2;
        
        % interpolation value
        ix = Nx/2;
        iy = Ny/2;
        
        a0 = 20;
        
        wavelength = 1000e-9;    % wavelength
        
        epoch = 10;              % we want 30 epochs
        images_per_epoch = 100; % we want 100 images per training session (epoch)
        
        distance_1 = 100e-2;      % propagation distance
        distance_2 = 50e-2;
        
        eta = 50.0;               % learning rate

% create a plate to detect digits
plate = detector_plate(Nx, Ny, nx, ny, nx/4, nx/20);

% load the mnist data into a MxM matrix format
data  = read_MNIST('training/images', 'training/labels');
test  = read_MNIST('testing/images', 'testing/labels');

% get the interpolation value k
kx = log2(double(ix - data.n_cols)/double(data.n_cols - 1))+1;
ky = log2(double(iy - data.n_rows)/double(data.n_rows - 1))+1;

% get the lowest interpolation value
k = min(kx, ky);

% we want to initalize the kernel mask with random phase and amplitude
kernel = internal_random_amp(Nx, Ny);
        
% generate the data to train on 
batch = v_batchwrapper;
batch.batch = get_batch(data, images_per_epoch);
superbatches(epoch) = batch;    %   a superbatch consists of [a, b, c...d]
                                %   where each are v_batchwrappers
                                %   each v_batchwrappers contains an 1xM
                                %   vector of of batches
for i=1:1:epoch-1
    batch = v_batchwrapper;
    batch.batch = get_batch(data, images_per_epoch);
    superbatches(i) = batch;
end

                                                                                                                    
% to store data generated per training session
dhs(images_per_epoch) = data_handler;

% iterate through all training session
for i=1:1:epoch
    batches = superbatches(i);

    disp("Starting training...");
    
    %
    % loop to go through each image per training session
    correct_per_epoch = 0;
    for j=1:1:images_per_epoch

        % the bottom below represents the forward pass
        batch        = batches.batch(j);

        img   = batch.img;
        label = batch.label;

        soln     = circle_at(Nx, Ny, nx, ny, nx/4, 0, nx/20);
        soln     = imrotate(soln, 36*label, 'crop');

        % normalize and resize the images
        nimg  = mask_resize(interp2(img.normalize(), k), Nx, Ny);

        % apply the kernel
        img_kernel  = nimg .* kernel;

        % propagate a distance
        img_prop_1  = propagate(img_kernel, distance_1, wavelength, Nx, Ny, nx, ny);

        % nonlinearity
        img_non     = nonlinear_forward (img_prop_1, a0);

        % propagate a second distance
        img_prop_2 = propagate(img_non, distance_2, wavelength, Nx, Ny, nx, ny);

        % pass through detector
        img_det     = img_prop_2 .* plate;
        img_det_mag = abs(img_det).^2;

        % store the results to be used in backpropagation
        dh = data_handler;
        dh.input_img      = nimg;
        dh.kernel_img     = img_kernel;
        dh.distance_1_img = img_prop_1;
        dh.nonlinear_img   = img_non;
        dh.distance_2_img = img_prop_2;
        dh.result_img     = img_det;
        dh.soln_img       = soln;
        dh.given_label    = label;
        dh.result_label   = detector_location(img_det_mag, Nx, Ny, nx, ny, nx/4, nx/20);

        dhs(j) = dh;
    end

    disp("Starting updating kernel...");

    %
    % start backpropagation for each epoch,
    % after back propagation, update the kernel mask
    for j=1:1:images_per_epoch
        handle = dhs(j);        % get the handle of each propagation

        % data
        i0 = handle.input_img;
        i1 = handle.kernel_img;
        i2 = handle.distance_1_img;
        i3 = handle.nonlinear_img;
        i4 = handle.distance_2_img;
        D  = handle.result_img;
        S  = handle.soln_img;

        m  = plate;

        %
        %   Let the following structure define the chain
        %   K  -> D1 -> N  -> D2 -> M  -> D
        %   R1 -> R2 -> R3 -> R4 -> R5
        %   
        
        d1      = get_propagation_distance(Nx, Ny, nx, ny, distance_1, wavelength);
        d2      = get_propagation_distance(Nx, Ny, nx, ny, distance_2, wavelength);

        dD      = D - S;
        dD_di4  = m;
        dD_di3  = apply_freq_mask(flip_180(d2), dD_di4);
        di3_di2 = nonlinear_backward(i2, a0);
        di3_di1 = apply_freq_mask(flip_180(d1), di3_di2);
        di1_dk  = i1;

        % compute backpropagation to change the kernel
        % all other parameters such as distance_1 and distance_2
        % are to be fixed.
        dD_dk   = dD .* dD_di3 .* di3_di1 .* di1_dk;

        handle.nabla = dD_dk;   %
        dhs(j) = handle;        % restore into handlers
                                % the handle now contains the nabla
    end

    %
    % update the kernel !!!

    new_kernel = zeros(Ny, Nx);
    for j=1:1:images_per_epoch
        handle = dhs(j);
        new_kernel = new_kernel + handle.nabla;
    end

    new_kernel = new_kernel * (eta/images_per_epoch);
    new_kernel = kernel - new_kernel;
    kernel     = new_kernel;

    disp("Starting testing...");

    testbatch = superbatches(1).batch;
    for j=1:1:length(testbatch)
        batch = testbatch(j);

        img   = batch.img;
        label = batch.label;

        % normalize and resize the images
        nimg  = mask_resize(interp2(img.normalize(), k), Nx, Ny);

        % apply the kernel
        img_kernel  = nimg .* kernel;

        % propagate a distance
        img_prop_1  = propagate(img_kernel, distance_1, wavelength, Nx, Ny, nx, ny);

        % nonlinearity
        img_non     = nonlinear_forward (img_prop_1, a0);

        % propagate a second distance
        img_prop_2 = propagate(img_non, distance_2, wavelength, Nx, Ny, nx, ny);

        % pass through detector
        img_det     = img_prop_2 .* plate;
        img_det_mag = abs(img_det).^2;

        result = detector_location(img_det_mag, Nx, Ny, nx, ny, nx/4, nx/20);

        if (result == label)
            correct_per_epoch = correct_per_epoch + 1;
        end
        
    end

    disp("@ Epoch="+i+", there was "+100*correct_per_epoch/images_per_epoch+"% correct.");
end



