function M = getFeatureMatrix( MILdataset, sigma )
%% Create feature matrix
    data = getdata(MILdataset);
    nBags = max(struct(MILdataset).ident.milbag);
    M = zeros(nBags, size(data,1));
    for i=1:nBags
        instances = findident(MILdataset,i, 'milbag');
        bag = data(instances,:);
        M(i,:) = bagembed(bag,data,sigma);
    end
end

