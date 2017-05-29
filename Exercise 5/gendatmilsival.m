function [MILdataset, labels] = gendatmilsival()
% creates a MIL dataset using the apples and banana images.

%% Settings
width = 30;
scale = 0.2;
dataset_dir = 'C:\Users\Milan\git\courses\ML\ML-Course\Exercise 5\sival_apple_banana\sival_apple_banana';
apple_label = 1;
banana_label = 2;

%% Load data
apple_path = strcat(dataset_dir, '\apple');
banana_path = strcat(dataset_dir, '\banana');
appleImgs = openImages(apple_path, scale);
bananaImgs = openImages(banana_path, scale);

nApples = size(appleImgs,1);
nBananas = size(bananaImgs,1);
nImgs = nApples + nBananas;

bags = cell(nImgs,1);   
labels = uint8(ones(nImgs,1)*apple_label);
labels(nApples+1:nImgs) = banana_label;

%% Extract Instances
for i=1:nApples
   bags{i} = extractInstances(squeeze(appleImgs(i,:,:,:)), width);
end
for i=1:nBananas
   bags{nApples+i} = extractInstances(squeeze(bananaImgs(i,:,:,:)),width); 
end

MILdataset = bags2dataset(bags, labels);
end

