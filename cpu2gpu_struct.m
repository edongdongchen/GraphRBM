% convert struct variables from
% gpu (jacket) to cpu
%---------------------------------------------------------------------------------------------------
% TNNLS-2018 paper: Graph regularized Restricted Boltzmann Machine
%---------------------------------------------------------------------------------------------------
%   Written by Dongdong Chen AT Sichuan University
%   Email: dongdongchen.scu@gmail.com

function A = cpu2gpu_struct(A, optdouble)

if ~exist('optdouble', 'var'),
    optdouble = 0;
end

fname = fieldnames(A);
for i = 1:length(fname),
    a = getfield(A, fname{i});
    if optdouble,
        A = setfield(A, fname{i}, gpuArray(double(a)));
    else
        A = setfield(A, fname{i}, gpuArray(single(a)));
    end
end

return;
