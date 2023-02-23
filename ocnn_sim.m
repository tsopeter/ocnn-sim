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
        
        wavelength = 1550e-9;    % wavelength
        
        epoch = 200;              % we want 200 epochs
        images_per_epoch = 150; % we want 150 images per training session (epoch)
        
        distance_1 = 50e-2;      % propagation distance
        distance_2 = 15e-2;
        
        eta1 = 1.2;              % learning rate
        eta2 = 1.2;              % 

        testing_ratio = 0.1;     % 10% of testing data (10k images)

        M_par_exec = 3;          % Number of cores for parallel execution

        P = 1;

disp("Getting data...");

% load the mnist data into a MxM matrix format
data  = read_MNIST('training/images', 'training/labels');
test  = read_MNIST('testing/images', 'testing/labels');

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
kernel = mask_resize(internal_random_amp(round(ix), round(iy)), Nx, Ny);

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

r1 = ratio_ix / 3.5;
r2 = ratio_iy / 25;

plate = detector_plate(size_d2_ix, size_d2_ix, ratio_ix, ratio_iy, r1, r2);

rd1  = rot90(d1, 2);
rd2  = rot90(d2, 2);

disp("Running first test...");

% run the test functions
initial_correct = test_a_batch(test_batch.batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, size_d2_ix, size_d2_iy, ratio_ix, ratio_iy, a0,  M_par_exec);

disp("Initially Correct: "+initial_correct+ " out of "+test_n_imgs);

% iterate through all training session

itj = 1:1:images_per_epoch;
dhs(images_per_epoch)= data_handler;
for i=1:1:epoch
    disp("@Epoch: "+i);

    disp("Starting training...");
    
    %
    % loop to go through each image per training session
    nabla_real = gpuArray(zeros(Ny, Nx, 'single'));
    nabla_imag = gpuArray(zeros(Ny, Nx, 'single'));

    batch = get_batch(data, images_per_epoch, 1);

    %
    % start training,
    % each training session involves a forward propagation pass
    % and a backward propagation pass.
    % Executes each pair per core.
    parfor (j=itj, M_par_exec)

        % the bottom below represents the forward pass
        btc = batch(j);
        
        dh         = forward_propagation(1, btc, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, size_d2_ix, size_d2_iy, ratio_ix, ratio_iy, a0);
        real_dh     = data_handler();
        imag_dh     = data_handler();

        real_dh.input_img      = real(dh.input_img);
        imag_dh.input_img      = imag(dh.input_img);

        real_dh.distance_1_img = real(dh.distance_1_img);
        imag_dh.distance_1_img = imag(dh.distance_1_img);

        real_dh.distance_2_img = real(dh.distance_2_img);
        imag_dh.distance_2_img = imag(dh.distance_2_img);

        real_dh.soln_img       = dh.soln_img;
        imag_dh.soln_img       = dh.soln_img;

        real_dh     = backward_propagation(1, real_dh, real(rd1), real(rd2), a0, P);
        imag_dh     = backward_propagation(1, imag_dh, imag(rd1), imag(rd2), a0, P);

        nabla_real  = nabla_real + real_dh.nabla;
        nabla_imag  = nabla_imag + imag_dh.nabla;
    end

    disp("Starting updating kernel...");

    a_nabla  = nabla_real * (eta1/images_per_epoch);
    b_nabla  = nabla_imag * (eta2/images_per_epoch);
    k_nabla  = a_nabla + 1i * b_nabla;

    kernel   = kernel - k_nabla;
    % kernel   = kernelLimit(kernel);
    % at every 5 epochs, run tests
    if (mod(i, 5) == 0)
        disp("Starting testing...");
        correct_per_epoch = test_a_batch(test_batch.batch, plate, kernel, d1, d2, Nx, Ny, nx, ny, r1, r2, k, size_d2_ix, size_d2_iy, ratio_ix, ratio_iy, a0, M_par_exec);
        disp("@ Epoch="+i+", there was "+correct_per_epoch+" out of "+test_n_imgs);
    end
end



