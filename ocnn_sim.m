% optical artifical neural network
% 
%

clear;

% define the parameters of the network

        Nx = 512;      % number of columns
        Ny = 512;      % number of rows
        
        % this defines the size of the display
        nx = 20e-3;
        ny = 20e-3;
        
        % interpolation value
        ix = Nx/2;
        iy = Ny/2;
        
        a0 = 20;
        
        wavelength = 1000e-9;    % wavelength
        
        epoch = 50;              % we want 50 epochs
        images_per_epoch = 10; % we want 10 images per training session (epoch)
        
        distance_1 = 30e-2;      % propagation distance
        distance_2 = 15e-2;
        
        eta = 0.05;               % learning rate

        testing_ratio = 0.1;     % 10% of testing data (10k images)

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

% create a batch to operate testing on
test_batch = v_batchwrapper;
test_batch.batch = get_batch(test, test.n_images*testing_ratio);

                                                                                                                    
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
        batch = batches.batch(j);
        dh    = forward_propagation(batch, kernel, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, nx/4, nx/20, k, a0);
        dhs(j) = dh;
    end

    disp("Starting updating kernel...");

    %
    % start backpropagation for each epoch,
    % after back propagation, update the kernel mask
    for j=1:1:images_per_epoch
        handle = dhs(j);        % get the handle of each propagation
        handle = backward_propagation(handle, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, a0);
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

    for j=1:1:test.n_images
        batch  = test_batch.batch(j);
        handle = forward_propagation(batch, kernel, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, nx/4, nx/20, k, a0);
        if (handle.result_label == handle.given_label)
            correct_per_epoch = correct_per_epoch + 1;
        end
        
    end

    disp("@ Epoch="+i+", there was "+100*correct_per_epoch/images_per_epoch+"% correct.");
end



