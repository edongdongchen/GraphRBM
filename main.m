% simple example
%---------------------------------------------------------------------------------------------------
% TNNLS-2018 paper: Graph regularized Restricted Boltzmann Machine
%---------------------------------------------------------------------------------------------------
%   Written by Dongdong Chen AT Sichuan University
%   Email: dongdongchen.scu@gmail.com

dataset = 'MNIST'; %YTF_CLUSTERING_LBP, COIL20, YaleB, MNIST
[x, y, ~, ~] = DataGen(dataset); % use your dataset by setting the value for x (data) and y (label, optional).

assert(isfloat(x), 'x must be a float');
assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');

[nSmp, nDim]= size(x);


nHid = 100;
filename = strcat(dataset, '_graph_nknn_',num2str(0),'.mat');
load(filename, 'phi');  % load pre-defined graph; replace the graph matrix phi with your defined graph which should of the size nSmp x nSmp

opts.GraphPhi = phi;
opts.lambda = 1e-2;     % graph regularization
opts.lrnrate = 1e-2;    % learning rate 1e-2
opts.l2reg = 1e-4;      % weight decay
opts.doGraphReg = 1;    % 1: GraphRBM; 0: RBM
opts.useGPU = 1; 		% 1: using GPU acceleration
opts.usepcd  = 1;
opts.batchsize = 100;
opts.numepochs = 100;
opts.momentum = 1e-8;

opts.visLearning = 0;   % 0: no visulization during training
opts.uiindex = [1,10, 220,221]; % samples index for visualization
opts.uirow = 28;
opts.uicol = 28;

disp('start GraphRBM training...');
[rbm, H] = train_GraphRBM_bin_bin(x, nHid, opts); % rbm: trained model, H: learned Hidden representation of data

savefilename = strcat('RBM_', dataset ,datestr(now, 'yyyymmddTHHMMSS'), 'mat');
save(savefilename, 'rbm', 'H');
