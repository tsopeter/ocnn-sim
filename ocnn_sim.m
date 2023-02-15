% optical artifical neural network
% 
%

clear;

% just as a reminder
% colormap('hot');
% imagesc(abs(kernel));
% colorbar('hot');

% define the parameters of the network

        Nx = 512;      % number of columns
        Ny = 512;      % number of rows
        
        % this defines the size of the display
        nx = 6e-2;
        ny = 6e-2;
        
        % interpolation value
        ix = Nx/2;
        iy = Ny/2;
        
        a0 = 20;
        
        wavelength = 1000e-9;    % wavelength
        
        epoch = 50;              % we want 50 epochs
        images_per_epoch = 500; % we want 500 images per training session (epoch)
        
        distance_1 = 30e-2;      % propagation distance
        distance_2 = 15e-2;
        
        eta = 3.0;               % learning rate

        testing_ratio = 0.1;     % 1% of testing data (10k images)

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
test_n_imgs = test.n_images * testing_ratio;

% clear unused data for reducing memory reqeuirements
data = [];
test = [];

% iterate through all training session

itj = 1:1:images_per_epoch;

for i=1:1:epoch

    disp("@Epoch: "+i);

    batches = superbatches(i);

    disp("Starting training...");
    
    %
    % loop to go through each image per training session
    nabla = zeros(Ny, Nx);
    for j=itj

        % the bottom below represents the forward pass
        batch  = batches.batch(j);
        dh     = forward_propagation(batch, kernel, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, nx/4, nx/20, k, a0);
        dh     = backward_propagation(dh, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, a0);
        nabla  = nabla + dh.nabla;
    end

    disp("Starting updating kernel...");

    %
    % start backpropagation for each epoch,
    % after back propagation, update the kernel mask

    nabla  = kernel - (nabla * (eta/images_per_epoch));
    kernel = nabla;

    disp("Starting testing...");

    correct_per_epoch = test_a_batch(test_batch, kernel, plate, distance_1, distance_2, wavelength, Nx, Ny, nx, ny, nx/4, nx/20, k, a0);
    disp("@ Epoch="+i+", there was "+(correct_per_epoch/test_n_imgs)*100.0+"% correct.");
end



