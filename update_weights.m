% update the parameters of rbm/graphrbm.
%   Feb/2016
%   Written by Dongdong Chen AT Sichuan University
%   Email: dongdongchen.scu@gmail.com

function [rbm, grad] = update_weights(rbm, grad, pos, neg, momentum, lrnrate, usepcd)

fname = fieldnames(rbm);

for i = 1:length(fname),
    % load fields
    pA = getfield(pos, fname{i});
    nA = getfield(neg, fname{i});
    gA = getfield(grad, fname{i});
    A = getfield(rbm, fname{i});
    
    if usepcd,
        gA = momentum*gA + (1-momentum)*lrnrate*pA;
        A = A + (gA - lrnrate*nA);
    else
        gA = momentum*gA + (1-momentum)*lrnrate*(pA - nA);
        A = A + gA;
    end
    
    % update accumulate parameter and weights
    grad = setfield(grad, fname{i}, gA);
    rbm = setfield(rbm, fname{i}, A);
end

return;
