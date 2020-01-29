% class_lbls --> Nx1 vector with class labels (1,...,C)
% clst_class_lbls --> Cx1 vector with the number of clusters per class
% clst_lbls --> Nx1 vector with the cluster labels (1,...,N1,1,...,N2,....,Nn)
function T = calculate_targets_singleview(class_lbls,clst_class_lbls, clst_lbls)
Label = unique(class_lbls);
T = [];
N = length(class_lbls);

nClasses = length(unique(class_lbls));
nClusters = length(unique(clst_lbls));
Y = randn(nClasses,nClasses-1);    Z = zeros(N,nClasses-1);

num2class = containers.Map(double(1),[1, 2]);
remove(num2class,1);
for class = 1:nClasses
    l = sum(class_lbls == class);
    if isKey(num2class,l)
        num2class(l) = [num2class(l) class];
    else
        num2class(l) = [class];
    end
end

keys = cell2mat(num2class.keys);
[sortedKeys, sortIdx] = sort( keys );

for ii=1:nClasses
    ind_i = find(class_lbls==Label(ii));     
    Z(ind_i,:) = repmat(Y(ii,:),length(ind_i),1); 
end
% Z(:,1) = [];
T = [T Z];

for l = sortedKeys
    classes = num2class(l);
    rep = length(classes);    

    RandVals = randn(rep*nClusters, rep*(nClusters - 1));
    i = 1;
    Tgen = zeros(N,rep*(nClusters - 1));
    for c = classes
        for clust = 1:clst_class_lbls(c)
                idxs = find(class_lbls == c & clst_lbls == clust);
                len = length(idxs);
                Tgen(idxs, :) = repmat(RandVals(i,:), len, 1);
                i = i+1;
        end
    end
    T = [T Tgen];
end
    
T = [ones(N,1) T]; 
[Y,R] = qr(T,0);  
Y(:,1) = []; 
T = Y';
end