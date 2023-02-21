% optical artifical neural network
% 
%

% just as a reminder
% colormap('hot');
% imagesc(abs(kernel));
% colorbar('hot');

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
        
        epoch = 120;              % we want 100 epochs
        images_per_epoch = 300; % we want 500 images per training session (epoch)
        
        distance_1 = 30e-2;      % propagation distance
        distance_2 = 15e-2;
        
        eta = 8.0;              % learning rate

        testing_ratio = 0.05;     % 10% of testing data (10k images)

        M_par_exec = 8;          % Number of cores for parallel execution

        P = 1;

        r1 = nx/5;
        r2 = nx/50;

% create a plate to detect digits
plate = detector_plate(Nx, Ny, nx, ny, r1, r2);

disp("Getting data...");

% load the mnist data into a MxM matrix format
data  = read_MNIST('training/images', 'training/labels');
test  = read_MNIST('testing/images', 'testing/labels');

% get the interpolation value k
kx = log2(double(ix - data.n_cols)/double(data.n_cols - 1))+1;
ky = log2(double(iy - data.n_rows)/double(data.n_rows - 1))+1;

% get the lowest interpolation value
k = min(kx, ky);

disp("Generating random kernel...");
% we want to initalize the kernel mask with random phase and amplitude
% if the kernel exists, uncomment the following
% kernel = load('data/kernel.mat').kernel;
kernel = mask_resize(internal_random_amp(round(ix), round(iy)), Nx, Ny);
        
% generate the data to train on 
batch = v_batchwrapper;
batch.batch = get_batch(data, images_per_epoch, 1);
superbatches(epoch) = batch;    %   a superbatch consists of [a, b, c...d]
                                %   where each are v_batchwrappers
                                %   each v_batchwrappers contains an 1xM
                                %   vector of of batches

disp("Generating batches...");
for i=1:1:epoch-1
    batch = v_batchwrapper;
    batch.batch = get_batch(data, images_per_epoch, 1);
    superbatches(i) = batch;
end

disp("Generating test bach...");
% create a batch to operate testing on
test_batch = v_batchwrapper;
test_batch.batch = get_batch(test, test.n_images*testing_ratio, 0);
test_n_imgs = test.n_images * testing_ratio;

% clear unused data for reducing memory reqeuirements

d1   = get_propagation_distance(Nx, Ny, nx, ny, distance_1 ,wavelength);
d2   = get_propagation_distance(Nx, Ny, nx, ny, distance_2, wavelength);

disp("Running first test...");

% run the test functions
initial_correct = test_a_batch(test_batch.batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, a0, M_par_exec);

disp("Initially Correct: "+initial_correct+ " out of "+test_n_imgs);

% iterate through all training session

itj = 1:1:images_per_epoch;
dhs(images_per_epoch)= data_handler;
g_batches = [];
for i=1:1:epoch
    disp("@Epoch: "+i);

    batches = superbatches(i);

    disp("Starting training...");

    % get the batches
    g_batches = batches.batch;
    
    %
    % loop to go through each image per training session
    nabla = zeros(Ny, Nx);
    parfor (j=itj, M_par_exec)

        % the bottom below represents the forward pass
        batch  = g_batches(j);
        dh     = forward_propagation(batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, a0);
        dh     = backward_propagation(dh, d1, d2, a0, P);
        nabla  = nabla + dh.nabla;
    end

    disp("Starting updating kernel...");

    %
    % start backpropagation for each epoch,
    % after back propagation, update the kernel mask

    a_nabla  = angle(nabla) * (eta/images_per_epoch);
    % b_nabla  = abs(nabla) * (eta/images_per_epoch);

    a_kernel = abs(kernel) .* exp(-1i * (a_nabla - angle(kernel)));
    kernel   = a_kernel;

    % at every 5 epochs, run tests
    if (mod(i, 5) == 0)
        disp("Starting testing...");
        correct_per_epoch = test_a_batch(test_batch.batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, a0, M_par_exec);
        disp("@ Epoch="+i+", there was "+correct_per_epoch+" out of "+test_n_imgs);
    end
end



