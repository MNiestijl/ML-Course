function plotInstances( instances, labels )
% It is assumed that there are 2 classes, and that the feature space is 3
% dimensional.

classes = unique(labels);
x1 = instances(labels==classes(1),:);
x2 = instances(labels==classes(2),:);

figure(1)
hold on;
scatter3(x1(:,1), x1(:,2), x1(:,3),'o', 'LineWidth', 2, 'MarkerEdgeColor', 'blue');
scatter3(x2(:,1), x2(:,2), x2(:,3),'o', 'LineWidth', 2, 'MarkerEdgeColor', 'red');
hold off
legend('1', '2');

end

