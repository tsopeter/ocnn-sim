% optical artifical neural network
% 
%

% just as a reminder
% colormap('hot');
% imagesc(abs(kernel));
% colorbar('hot');

% define the parameters of the network

        Nx = 1024;      % number of columns
        Ny = 1024;      % number of rows

        inputSize  = [Ny, Nx, 1];
        numClasses = 10;    % one for each digit 0-9
        
        % this defines the size of the display
        nx = 40e-3;
        ny = 40e-3;

        % filter_1 ratio
        ratio=4;
        
        % interpolation value
        ix = Nx/ratio;
        iy = Ny/ratio;
        
        a0 = 20;
        
        wavelength = 1000e-9;    % wavelength
        
        epoch = 120;              % we want 100 epochs
        images_per_epoch = 250; % we want 500 images per training session (epoch)
        
        distance_1 = 30e-2;      % propagation distance
        distance_2 = 15e-2;
        
        eta = 12.0;              % learning rate

        testing_ratio = 0.1;     % 10% of testing data (10k images)

        P = 1;

        r1 = nx/7;
        r2 = nx/50;

% create a plate to detect digits
plate = detector_plate(Nx, Ny, nx, ny, r1, r2);

disp("Getting data...");

oldPath       = addpath(fullfile("~/Documents/GitHub/ocnn-sim/Images/"));
dataimagefile = 'Images/train-images-idx3-ubyte.gz';
datalabelfile = 'Images/train-labels-idx1-ubyte.gz';
testimagefile = 'Images/t10k-images-idx3-ubyte.gz';
testlabelfile = 'Images/t10k-labels-idx1-ubyte.gz';

XTrain = processImagesMNIST(dataimagefile);
YTrain = processLabelsMNIST(datalabelfile);
XTest = processImagesMNIST(testimagefile);
YTest = processLabelsMNIST(testlabelfile);

cols = length(XTrain(:,:,1));
rows = length(XTrain(:,:,1).');

% get the interpolation value k
kx = log2(double(ix - cols)/double(cols - 1))+1;
ky = log2(double(iy - rows)/double(rows - 1))+1;

% get the lowest interpolation value
k = min(kx, ky);

% restore the path
addpath(oldPath);

d1   = get_propagation_distance(round(ix), round(iy), nx/ratio, ny/ratio, distance_1, wavelength);

ILayer    = imageInputLayer(inputSize, 'Name', 'input_layer'); 
in_KLayer = imageInputLayer(inputSize, 'Name', 'kernel_input');
KLayer  = multiplicationLayer(2, 'Name', 'kernel_layer');
D1Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding','same', 'Name', 'prop_layer_1');
NLayer  = functionLayer(@(x)sa_forward(x), 'Name', 'nonlinear_layer');
D2Layer = convolution2dLayer([round(ix), round(iy)], 1, 'Stride', 1, 'Padding', 'same', 'Name', 'prop_layer_2');
RLayer  = multiplicationLayer(2, 'Name', 'plate_layer');
in_RLayer = imageInputLayer(inputSize, 'Name', 'plate_input');

lgraph = layerGraph();
lgraph = addLayers(lgraph, ILayer);
lgraph = addLayers(lgraph, KLayer);
lgraph = addLayers(lgraph, in_KLayer);
lgraph = addLayers(lgraph, D1Layer);
lgraph = addLayers(lgraph, NLayer);
lgraph = addLayers(lgraph, D2Layer);
lgraph = addLayers(lgraph, RLayer);
lgraph = addLayers(lgraph, in_RLayer);

lgraph = connectLayers(lgraph, 'input_layer', 'kernel_layer/in1');
lgraph = connectLayers(lgraph, 'kernel_input', 'kernel_layer/in2');
lgraph = connectLayers(lgraph, 'kernel_layer', 'prop_layer_1/in');
lgraph = connectLayers(lgraph, 'prop_layer_1', 'nonlinear_layer/in');
lgraph = connectLayers(lgraph, 'nonlinear_layer', 'prop_layer_2/in');
lgraph = connectLayers(lgraph, 'prop_layer_2', 'plate_layer/in1');
lgraph = connectLayers(lgraph, 'plate_input', 'plate_layer/in2');

plot(lgraph);


