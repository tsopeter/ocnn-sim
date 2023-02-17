% optical artifical neural network
% 
%

% just as a reminder
% colormap('hot');
% imagesc(abs(kernel));
% colorbar('hot');

% define the parameters of the network

        Nx = 256;      % number of columns
        Ny = 256;      % number of rows
        
        % this defines the size of the display
        nx = 6e-2;
        ny = 6e-2;
        
        % interpolation value
        ix = floor(Nx/2);
        iy = floor(Ny/2);
        
        a0 = 30;
        
        wavelength = 1000e-9;    % wavelength
        
        epoch = 120;              % we want 120 epochs
        images_per_epoch = 100; % we want 100 images per training session (epoch)
        
        distance_1 = 15e-2;      % propagation distance
        distance_2 = 15e-2;
        
        eta = 0.1;              % learning rate

        testing_ratio = 0.01;     % 10% of testing data (10k images)

        M_par_exec = 14;          % Number of cores for parallel execution

        n_bars = 9.25;

% create a plate to detect digits
plate = mask_resize(new_detector_plate(round(ix/2), round(iy/2), n_bars), Nx, Ny);

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
kernel = mask_resize(internal_random_amp(ix, iy), Nx, Ny);

disp("Generating test bach...");
% create a batch to operate testing on
test_batch = v_batchwrapper;
test_batch.batch = get_batch(test, test.n_images*testing_ratio, 0);
test_n_imgs = test.n_images * testing_ratio;

d1   = get_propagation_distance(Nx, Ny, nx, ny, distance_1 ,wavelength);
d2   = get_propagation_distance(Nx, Ny, nx, ny, distance_2, wavelength);

disp("Running first test...");

% run the test functions
initial_correct = testing(test_batch.batch, plate, kernel, d1, d2, Nx, Ny, k, a0, n_bars, ix, iy, M_par_exec);

disp("Initially Correct: "+initial_correct+" out of "+test_n_imgs);

% iterate through all training session

itj = 1:1:images_per_epoch;
dhs(images_per_epoch)= data_handler;
for i=1:1:epoch
    disp("@Epoch: "+i);

    disp("Starting training...");

    % generate a batch
    batches = get_batch(data, images_per_epoch, 1);
    
    %
    % loop to go through each image per training session
    nabla = zeros(Ny, Nx);
    parfor (j=itj, M_par_exec)

        % the bottom below represents the forward pass
        batch  = batches(j);
        dh     = forward_propagation(batch, plate, kernel, d1, d2, Nx, Ny, k, a0, n_bars);
        dh     = backward_propagation(dh, d1, d2, a0);
        nabla  = nabla + dh.nabla;
    end

    disp("Starting updating kernel...");

    %
    % start backpropagation for each epoch,
    % after back propagation, update the kernel mask

    nabla  = (eta/images_per_epoch) * angle(nabla);
    kernel = kernel .* exp(1i * nabla);

    % at every 5 epochs, run tests
    if (mod(i, 5) == 0)
        disp("Starting testing...");
        correct_per_epoch = testing(test_batch.batch, plate, kernel, d1, d2, Nx, Ny, k, a0, n_bars, ix, iy, M_par_exec);
        disp("@ Epoch="+i+", there was "+correct_per_epoch+" out of "+test_n_imgs);
    end
end



