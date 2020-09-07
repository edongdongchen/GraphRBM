function [rbm, currentH] = train_GraphRBM_bin_bin(x, nHid, opts)
% Graph regularized Restricted Boltzmann Machine (GraphRBM, binary input&output).
% With GPU Acceleration & Optimization (optional)
% input:
% x           -   nSmp x nDim data matrix (training data)
% nHid     -   1x1  dimension of hidden layer
% opts      -   parameters
%
% output:
% rbm           -  struct
% currentH  -  representation of x
%---------------------------------------------------------------------------------------------------
% TNNLS-2018 paper: Graph regularized Restricted Boltzmann Machine
%---------------------------------------------------------------------------------------------------
%   version 2.0 --Oct/2016
%   version 1.0 --Feb/2016
%
%   Written by Dongdong Chen AT Sichuan University
%   Email: dongdongchen.scu@gmail.com
    
    [nSmp, nDim ]= size(x);
    nV  = nDim;
    nH  = nHid;
    numbatches = floor(nSmp / opts.batchsize);
    
    % INITIALIZATION
    rbm = [];
    rbm.W = randn(nH, nV)*0.01;
    rbm.b     = arcsigm(clip(mean(x, 1)))';
    rbm.c     = zeros(nH, 1);
    
    % convert to gpu variables
    if opts.useGPU,
        rbm = cpu2gpu_struct(rbm);
    end
    % structs for gradients
    grad = replicate_struct(rbm, 0);
    pos   = replicate_struct(rbm, 0);
    neg   = replicate_struct(rbm, 0);

    phi = opts.GraphPhi;
    currentH = zeros(nSmp, nH);

    if opts.useGPU,
        rbm = cpu2gpu_struct(rbm);
    end

    history = [];
    history.recon_err = [];

    part1 = zeros(nSmp, nH);
    part2 = zeros(nSmp, nH);
    recon_data = zeros(size(x));

    % START LEARNING +++++++++++++++++++++++++++++++++++++++++++++++++
    for i = 1 : opts.numepochs % i: epoch
        randidx = randperm(nSmp);
        recon_err_epoch = zeros(1, numbatches);

        for j = 1 : numbatches    % j: batch
            batchidx = randidx((j - 1) * opts.batchsize + 1 : j * opts.batchsize);
            batch_data = x(batchidx, :);
            batch_phi   = phi(batchidx, :);
            delta = zeros(opts.batchsize, nH);

            if opts.useGPU,
                batch_data = gpuArray(single(batch_data));
                if i > 1 && opts.doGraphReg
                    dump = repmat(sum(batch_phi, 2), 1, nH) - 2 * batch_phi * currentH;  
                    delta = gpuArray(dump);  
                end
            else
                if i > 1 && opts.doGraphReg
                    delta = repmat(sum(batch_phi, 2), 1, nH) - 2 * batch_phi * currentH;
                end
            end

            % positive
            v1_state = batch_data;

            %+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
            aa = repmat(rbm.c', opts.batchsize, 1) + v1_state * rbm.W';
            bb = opts.lambda * delta;
            part1(batchidx, :) = gather(aa);
            part2(batchidx, :) = gather(bb);

            %+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

            h1_prop = sigmoid(aa - bb);
            h1_state = double(h1_prop > rand(size(h1_prop)));

            % negative
            v2_prop = sigmoid(repmat(rbm.b', opts.batchsize, 1) + h1_state * rbm.W);
            v2_state = double(v2_prop > rand(size(v2_prop)));
            h2_prop = sigmoid(repmat(rbm.c', opts.batchsize, 1) + v2_state * rbm.W'   - opts.lambda * delta);
            h2_state = double(h2_prop>rand(size(h2_prop)));

            recon = sigmoid(repmat(rbm.b', opts.batchsize, 1) + h1_prop * rbm.W);
            recon_err_epoch(j) = gather(sum(sum(v1_state - recon).^2)) / opts.batchsize;

            %+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
            recon_data(batchidx, :) = gather(recon);
            %+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

            pos.W = h1_prop' * v1_state / opts.batchsize  - opts.l2reg*rbm.W; %<vh>_data
            pos.b  = mean(v1_state, 1)';                                %<v>_data
            pos.c  = mean(h1_prop, 1)';                                %<h>_data

            neg.W = h2_state' * v2_prop / opts.batchsize; %<vh>_model
            neg.b = mean(v2_prop, 1)';                                 %<v>_model
            neg.c = mean(h2_state, 1)';                                 %<h>_model

            [rbm, grad] = update_weights(rbm, grad, pos, neg, opts.momentum, opts.lrnrate, opts.usepcd);

            currentH(batchidx, :) = gather(h1_prop);
            %currentH(batchidx, :) = gather(sigm(repmat(rbm.c', opts.batchsize, 1) + batch_data * rbm.W' - opts.lambda * delta));
        end
        history.recon_err = [history.recon_err, gather(sum(recon_err_epoch))/numbatches];

        if opts.visLearning
            %index = [2, 22];%
            index = opts.uiindex;
            ShowLearning(1, i, part1, -part2, x, currentH, recon_data, index, history.recon_err, opts.uirow, opts.uicol);
            if i ==10%opts.numepochs
                %ShowSingleLearning(100, part1, -part2, x, double(currentH>rand(size(currentH))), recon_data, index, history.recon_err, opts.uirow, opts.uicol);
                ShowSingleLearning(100, part1, -part2, x, currentH, recon_data, index, history.recon_err, opts.uirow, opts.uicol);
            end
            %pause(0.01);
        end

        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(history.recon_err(i))]);
    end
end
