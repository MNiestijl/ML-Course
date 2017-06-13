classdef KNNmil
    % KNN object designed for MIL problems.
    
    properties
        k=1;
        dist = @(A,B) hausdorff(A,B);
        data;
        labels;
        classes;
    end

    methods
        
        % Set distance function if provided
        %   1: k (default=1)  
        %   2: distance function to use. Either 'minimal' or 'hausdorff'.
        function obj = KNNmil(k,distance)
            obj.k = k;  
            if strcmp(distance, 'minimal')
                obj.dist = @(A,B) minDist(A,B);
            elseif strcmp(distance, 'maximal')
                obj.dist = @(A,B) maxDist(A,B);
            elseif strcmp(distance, 'hausdorff')
                obj.dist = @(A,B) hausdorffDist(A,B);
            end
        end

        % data: cell (bags×1) containing arrays (instances × features)       
        function obj = fit(obj,data,labels)    
            obj.data = data;
            obj.labels = labels;
            obj.classes = unique(labels);
        end

        % data: cell (bags×1) containing arrays (instances × features)
        function labels = predict(obj, data)
            labels = arrayfun(@(i) predictInstance(obj,data{i}), 1:size(data,1))';
        end

        function s = score(obj, data, labels)
            s = mean(obj.predict(data)==labels);
        end
    end

    methods (Access = private)
        
        function label = predictInstance(obj, instance)
            [neighbors, distances] = getNeighbors(obj, instance);
            label = getPrediction(obj,neighbors, distances);
        end
        
        % Return the indices of the k neighbors and the distance to them
        function [neighbors, distances] = getNeighbors(obj, instance)
            allDistances = arrayfun(@(i) obj.dist(obj.data{i},instance), 1:size(obj.data,1));
            [sorted, inds] = sort(allDistances);
            neighbors = inds(1:obj.k);
            distances = sorted(1:obj.k);
        end

        % Return the predicted label based on majority voting.
        % TODO: add weights based on distance.
        function label = getPrediction(obj,neighbors, distances)
            votes = uint8(zeros(length(obj.classes)));
            for i=1:length(obj.classes)
                votes(i) = sum(obj.labels(neighbors)==obj.classes(i));
            end
            [~,ix] = max(votes);
            label = obj.classes(ix(1));
        end
    end
end
