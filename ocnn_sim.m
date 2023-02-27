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
        nx = 80e-3;
        ny = 80e-3;
        
        % interpolation value

        ratio = 2;

        ix = Nx/ratio;
        iy = Ny/ratio;
        
        a0 = 20;
        
        wavelength = 1000e-9;    % wavelength
        
        epoch = 200;              % we want 100 epochs
        images_per_epoch = 16; % we want 16 images per training session (epoch)
        
        distance_1 = 50e-2;      % propagation distance
        distance_2 = 15e-2;
        
        eta = 0.05;              % learning rate

        testing_ratio = 0.1;     % 10% of testing data (10k images)

        M_par_exec = 2;          % Number of cores for parallel execution

        P = 0.5;

        read_MNIST_flag  = 0;     % zero the flag is already read!!
        load_KERNEL_flag = 0;     % zero the flag if kernel needs to be generated

disp("Getting data...");

% load the mnist data into a MxM matrix format
if (read_MNIST_flag == 1)
    data  = read_MNIST('training/images', 'training/labels');
    test  = read_MNIST('testing/images', 'testing/labels');
end

% get the interpolation value k
kx = log2(double(ix - data.n_cols)/double(data.n_cols - 1))+1;
ky = log2(double(iy - data.n_rows)/double(data.n_rows - 1))+1;

% get the lowest interpolation value
k = min(kx, ky);
k = k/2;

disp("Generating random kernel...");
% we want to initalize the kernel mask with random phase and amplitude
% if the kernel exists, uncomment the following
% kernel = load('data/kernel.mat').kernel;
if (load_KERNEL_flag == 1)
    kernel = mask_resize(internal_random_amp(round(ix), round(iy)), Nx, Ny);
else
    disp("Grabbing pre-computed kernel...");
    kernel = load('data\kernel.mat');
end

disp("Generating test bach...");
% create a batch to operate testing on
test_batch = v_batchwrapper;
test_batch.batch = get_batch(test, test.n_images*testing_ratio, 0);
test_n_imgs = test.n_images * testing_ratio;

% generating kernels
disp("Generating layer kernels...");

d1   = fftshift(get_propagation_distance(round(ix), round(iy), nx/ratio, ny/ratio, distance_1 ,wavelength));

% compute the necessary ratio for the next
size_d1_next = conv2(kernel, d1, 'valid');
size_d1_ix   = length(size_d1_next(:,1));
size_d1_iy   = length(size_d1_next(1,:));
ratio_ix     = size_d1_ix / ratio;
ratio_iy     = size_d1_iy / ratio;

d2   = fftshift(get_propagation_distance(round(size_d1_ix/ratio), round(size_d1_iy/ratio), ratio_ix, ratio_iy, distance_2, wavelength));

size_d2_next = conv2(size_d1_next, d2, 'valid');
size_d2_ix   = length(size_d2_next(:,1));
size_d2_iy   = length(size_d2_next(1,:));
ratio_ix     = (size_d2_ix / Nx) * nx;
ratio_iy     = (size_d2_iy / Ny) * ny;

% create a plate to detect digits
disp("Creating detector plate...");

r1 = ratio_ix / 4;
r2 = ratio_iy / 25;

% create the detector plate,
% the detector plate is used for detecting digits
plate = detector_plate(size_d2_ix, size_d2_ix, ratio_ix, ratio_iy, r1, r2);

% take the hermitian of the convolution kernels
% used for backpropagation
rd1  = d1';
rd2  = d2';

disp("Running first test...");

% run the test functions
initial_correct = test_a_batch(test_batch.batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, size_d2_ix, size_d2_iy, ratio_ix, ratio_iy, a0,  M_par_exec);

disp("Initially Correct: "+initial_correct+ " out of "+test_n_imgs);

% iterate through all training session

itj = 1:1:images_per_epoch;
g_batches = [];
for i=1:1:epoch
    disp("@Epoch: "+i);

    disp("Starting training...");
    
    %
    % loop to go through each image per training session
    nabla = gpuArray(zeros(Ny, Nx, 'single'));

    batches = get_batch(data, images_per_epoch, 1);
    parfor (j=itj, M_par_exec)

        % the bottom below represents the forward pass
        batch     = batches(j);
        dh        = forward_propagation(batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, size_d2_ix, size_d2_iy, ratio_ix, ratio_iy, a0);
        dh        = backward_propagation(dh, rd1, rd2, a0, P);
        nabla = nabla + dh.nabla;
    end

    disp("Starting updating kernel...");

    %
    % start backpropagation for each epoch,
    % after back propagation, update the kernel mask

    nabla  = nabla * (eta/images_per_epoch);
    kernel = kernel - nabla;

    % at every 5 epochs, run tests
    if (mod(i, 5) == 0)
        disp("Starting testing...");
        correct_per_epoch = test_a_batch(test_batch.batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, size_d2_ix, size_d2_iy, ratio_ix, ratio_iy, a0, M_par_exec);
        disp("@ Epoch="+i+", there was "+correct_per_epoch+" out of "+test_n_imgs);
    end
end



