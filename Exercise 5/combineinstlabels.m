function label = combineinstlabels(labels , rule, varargin)
% Supported rules:
% 'majority' - majority voting
% 'one' - at least one positive
    classes = unique(labels);

    %% Perform combining
    if length(classes)==1
        label=classes(1);
    elseif strcmp(rule, 'majority')
        if sum(labels==classes(1))< sum(labels==classes(2))
            label = classes(2);
        else
            label = classes(1);
        end

    elseif strcmp(rule, 'one')
        % Decide which class is regarded as 'positive'.
        if length(varargin)==1 && ismember(varargin{1},classes)
            positiveLabel = classes(varargin{1});
            negativeLabel = classes(classes~=varargin{1});
        else
            positiveLabel = classes(2);
            negativeLabel = classes(1);
        end

        % Evaluate the combiner
        if any(labels==positiveLabel)
            label = positiveLabel;
        else
            label = negativeLabel;
        end
    end

end

