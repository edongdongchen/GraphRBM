% -------------------------------------------------------------------------
% draw sample from bernoulli distribution x
%   Feb/2016
%   Written by Dongdong Chen AT Sichuan University
%   Email: dongdongchen.scu@gmail.com
% -------------------------------------------------------------------------

function y = sample_bernoulli(x, optgpu)

if ~exist('optgpu','var'),
    optgpu = 0;
end

y = x > rand(size(x));
if optgpu,
    y = gpuArray(single(y));
else
    y = double(y);
end

return;

