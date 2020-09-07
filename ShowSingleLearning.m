function ShowSingleLearning(fid, part1, part2, X, H, recon, index, err, row, col)

[nSmp, nDim] = size(H);

figure(fid);
clf;
set(gcf,'name','Graph regularized RBM');

ui_row = length(index);

gap = .15;
pSize = 3;
% data
figure(fid)
for i = 1 : ui_row
    subplottight(ui_row, 1, i, gap);
    imshow(reshape(X(index(i), :), row, col));
    if i == 1,  title('sample'); end
end
drawnow

% reconstruction
figure(fid+1)
for i = 1 : ui_row
    subplottight(ui_row, 1, i, gap);
    imshow(reshape(recon(index(i), :), row, col));
    if i == 1,  title('reconstruction'); end
end
drawnow


% Hidden representation
figure(fid+2)

for i = 1 : ui_row
    subplottight(ui_row, 1, i, gap);
    stem(1:nDim, H(index(i),:), 'fill','b-','MarkerSize',pSize);
    if i == 1,  title('h'); end
end
drawnow

% Part 1
figure(fid+3)
for i = 1 : ui_row
    subplottight(ui_row, 1, i, gap);
    stem(1:nDim, part1(index(i),:), 'fill','b-','MarkerSize',pSize);
    if i == 1,  title('W*v+c'); end
end
drawnow


% Part 2
figure(fid+4)
for i = 1 : ui_row
    subplottight(ui_row, 1, i, gap);
    stem(1:nDim, part2(index(i),:), 'fill','b-','MarkerSize',pSize);
    if i == 1,  title('\delta'); end
end
drawnow
