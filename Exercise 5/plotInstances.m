function plotInstances( MILdataset, labels, varargin )
set(0,'defaulttextinterpreter','latex')
% It is assumed that there are 2 classes, and that the feature space is 3
% dimensional.
% optinal input: indices of instances to be highlighted.
instances = getdata(MILdataset);
ident = struct(MILdataset).ident.ident;
milbag = struct(MILdataset).ident.milbag;
classes = unique(labels);
x1 = instances(ident(arrayfun(@(i) labels(i),milbag)==classes(1)),:);
x2 = instances(ident(arrayfun(@(i) labels(i),milbag)==classes(2)),:);
disp(length(x1)+length(x2))
disp(size(instances,1))

figure(1);
hold on;
scatter3(x1(:,1), x1(:,2), x1(:,3),'o', 'filled', 'MarkerEdgeColor', 1./255.*[54, 107, 13], 'MarkerFaceColor',  1./255.*[94, 147, 53]);
scatter3(x2(:,1), x2(:,2), x2(:,3),'o', 'filled', 'MarkerEdgeColor', 1./255.*[136, 47, 47], 'MarkerFaceColor',  1./255.*[186, 87, 87]);
if ~isempty(varargin)
   x3 = instances(varargin{1},:); 
   scatter3(x3(:,1), x3(:,2), x3(:,3),'o', 'LineWidth', 3, 'MarkerEdgeColor', 1./255.*[255, 191, 0]);
end
hold off
axis([0 255 0 255 0 255])
legend('Apple', 'Banana');
xlabel('Red','fontsize', 15);
ylabel('Green','fontsize', 15);
zlabel('Blue','fontsize', 15);
title('Scatterplot of the instances','fontsize', 20)
end

