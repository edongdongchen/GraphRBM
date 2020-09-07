function ShowLearning(fid, epoch, part1, part2, X, H, recon, index, err, row, col)

[nSmp, nDim] = size(H);

figure(fid);
clf;
set(gcf,'name',strcat('Graph regularized RBM', num2str(epoch)));

ui_row = length(index);
dot_size = 2;

for i = 1 : ui_row
    figure(fid)
    
    subplottight(ui_row, 5, (i-1)*5 + 1, .15);
    imshow(reshape(X(index(i), :), row, col));
    title('sample');
    
    subplottight(ui_row, 5, (i-1)*5 + 2, .15);
    imshow(reshape(recon(index(i), :), row, col));
    title('reconstruction');

    subplottight(ui_row, 5, (i-1)*5 + 3, .15);
    stem(1: nDim, H(index(i),:), 'fill','b-','MarkerSize',dot_size);
    title('h');

    subplottight(ui_row, 5, (i-1)*5 + 4, .15);
    stem(1: nDim, part1(index(i),:), 'fill','b-','MarkerSize',dot_size);
    title('W*v+c');

    subplottight(ui_row, 5, (i-1)*5 + 5, .15);
    stem(1: nDim, part2(index(i),:), 'fill','b-','MarkerSize',dot_size);
    title('\delta');
end

drawnow
