function features = extractInstances2( image, width, varargin)
% Segments an image using the Mean Shift algorithm, computes the average
% red, green and blue color per segment and returns the resulting features
% in a small data matrix
% if optional argument is provided, the segments are plotted.

    segments = uint8(im_meanshift(image, width));
    nSegments = max(max(segments));
    
    % get averages
    features = zeros(nSegments, 12);
    for i=1:nSegments
       features(i,:) = getSegmentStatistics(image, segments, i); 
    end

    %% Plot the segments
    if ~isempty(varargin)
        imshow(segments, [1,nSegments]);
        disp(nSegments);
    end
end

function statistics = getSegmentStatistics(image,segments,targetSegment)

    redIm = image(:,:,1);
    greenIm = image(:,:,2);
    blueIm = image(:,:,3);
    
    target = segments==targetSegment;
    
    red = mean(redIm(target));
    maxred = max(redIm(target));
    minred = min(redIm(target));
    stdred = std(double(redIm(target)));
    
    green = mean(greenIm(target));
    maxgreen = max(greenIm(target));
    mingreen = min(greenIm(target));
    stdgreen = std(double(greenIm(target)));
    
    blue = mean(blueIm(target));
    maxblue = max(blueIm(target));
    minblue = min(blueIm(target));
    stdblue = std(double(blueIm(target)));
    
    statistics = [  red, blue, green,...
                    maxred, maxblue, maxgreen,...
                    minred, minblue, mingreen, ...
                    stdred, stdblue, stdgreen];
end
