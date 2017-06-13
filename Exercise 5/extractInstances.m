function features = extractInstances( image, width, varargin)
% Segments an image using the Mean Shift algorithm, computes the average
% red, green and blue color per segment and returns the resulting features
% in a small data matrix
% if optional argument is provided, the segments are plotted.

    segments = uint8(im_meanshift(image, width));
    nSegments = max(max(segments));
    
    % get averages
    features = zeros(nSegments, 3);
    for i=1:nSegments
       features(i,:) = getSegmentAverages(image, segments, i); 
    end

    %% Plot the segments
    if ~isempty(varargin)
        imshow(segments, [1,nSegments]);
        disp(nSegments);
    end
end

function averages = getSegmentAverages(image,segments,targetSegment)

    redIm = image(:,:,1);
    greenIm = image(:,:,2);
    blueIm = image(:,:,3);
       
    red = mean(redIm(segments==targetSegment));
    green = mean(greenIm(segments==targetSegment));
    blue = mean(blueIm(segments==targetSegment));
    
    averages = [red, blue, green];
    if any(isnan(averages));
        fprintf('\naverages: %s', averages);
        fprintf('\ntargetSegment: %i', targetSegment);
        fprintf('\nsize segments: %i', size(segments));
        fprintf('\nsize redIm: %d', redIm);
        fprintf('\nsize greenIm: %d', greenIm);
        fprintf('\nsize blueIm: %d', blueIm);
        error('NaN encountered!')
    end
end
