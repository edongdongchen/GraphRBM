function ShowLearning(fid, epoch, part1, part2, X, H, recon, index, err, row, col)

[nSmp, nDim] = size(H);

figure(fid);
clf;
set(gcf,'name',strcat('Graph regularized RBM', num2str(epoch)));

ui_row = length(index);
dot_size = 2;
for i = 1 : ui_row
    figure(fid)
    
    %subplot(ui_row, 5, (i-1)*5 + 1)
    subplottight(ui_row, 5, (i-1)*5 + 1, .15);
    %aa = X(index(i), :);
    imshow(reshape(X(index(i), :), row, col));
    title('sample');
    %hold on;
    
	%subplot(ui_row, 5, (i-1)*5 + 2)
    subplottight(ui_row, 5, (i-1)*5 + 2, .15);
    imshow(reshape(recon(index(i), :), row, col));
    title('reconstruction');
    %hold on;

    %subplot(ui_row, 5, (i-1)*5 + 3)
    subplottight(ui_row, 5, (i-1)*5 + 3, .15);
    stem(1: nDim, H(index(i),:), 'fill','b-','MarkerSize',dot_size);
    title('h');
    %hold on;

    %subplot(ui_row, 5, (i-1)*5 + 4)
    subplottight(ui_row, 5, (i-1)*5 + 4, .15);
    stem(1: nDim, part1(index(i),:), 'fill','b-','MarkerSize',dot_size);
	title('W*v+c');
    %hold on;

    %subplot(ui_row, 5, (i-1)*5 + 5)
    subplottight(ui_row, 5, (i-1)*5 + 5, .15);
    stem(1: nDim, part2(index(i),:), 'fill','b-','MarkerSize',dot_size);
    title('\delta');
    %hold on;
    
%     subplottight(3,3,5,.15);
%     imagesc(RBM.aHid);
%     colorbar
%     title('Hidden Unit Activations');

% subplottight(3,3,7,.15);
% plot(RBM.auxVars.error);
% title('Reconstruction errors');
    
end
%hold off;

drawnow
% figure(fid+1)
% plot(1:length(err), err, 'r-');