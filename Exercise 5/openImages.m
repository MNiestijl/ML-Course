function images = openImages( folder, varargin )
% Open images from given directory. 
% Optional argument: Downsize scaling parameter
% parameter to downsize the images
% Supported formats: .jpg

% Add trailing '/' if necessary
if (folder(end) ~= '\') && (folder(end) ~= '/')
    folder = strcat(folder, '/');
end

if length(varargin)==1
    scale = varargin{1};
else
    scale = 1;
end

%% Read images
content = dir(strcat(folder,'*.jpg'));
nImgs = length(content);

% load first image to determine the size of the images
image = readImage(strcat(folder,content(1).name), scale);
[m,n,d] = size(image);
images = uint8(zeros(nImgs, m,n,d));
images(1,:,:,:) = image;

% load the rest of the images
for j=2:nImgs
    filename = content(j).name;
    images(j,:,:,:) = readImage(strcat(folder,filename), scale);
end


end

function image = readImage(filename, scale)
    image = imread(filename);
    if scale~=1
        image = imresize(image,scale);
    end
end

