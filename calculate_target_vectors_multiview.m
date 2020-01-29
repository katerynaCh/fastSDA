function [T] = calculate_target_vectors_multiview(class_lbls,clst_class_lbls,cluster_labels)

randn('seed',42);

Label = unique(class_lbls);
T = [];
N = length(class_lbls);
V = length(cluster_labels);
view_label = [];
class_lbls = repmat(class_lbls, V, 1);
clst_lbls = [];
for view =1:length(cluster_labels)
    clst_lbls = [clst_lbls; cluster_labels(int2str(view))];
    view_label = [view_label; view*ones(N,1)];
end
nClasses = length(unique(class_lbls));
nClusters = length(unique(clst_lbls));
Y = randn(nClasses,nClasses-1);    Z = zeros(N*V,nClasses-1);

for ii=1:nClasses
    ind_i = find(class_lbls==Label(ii)); 
    
    Z(ind_i,:) = repmat(Y(ii,:),length(ind_i),1); %create random value per label
end
% Z(:,1) = [];
T = [T Z];


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

for l = sortedKeys
    classes = num2class(l);
    rep = length(classes);
    

    RandVals = randn(rep*nClusters*V, rep*(V*nClusters - 1));
    i = 1;
    Tgen = zeros(V*N,rep*(V*nClusters - 1));
    for c = classes
        for clust = 1:clst_class_lbls(c)
            for v = 1:V
                idxs = find(class_lbls == c & clst_lbls == clust & view_label == v);
                len = length(idxs);
                Tgen(idxs, :) = repmat(RandVals(i,:), len, 1);
                i = i+1;
            end
        end
        
    end
    T = [T Tgen];
end
    
T = [ones(V*N,1) T]; %first one to 1s
[Y,R] = qr(T,0);   %decompose so that A = Q*R  R = mxn upper triangular matrix; Q = mxm unitary matrix
Y(:,1) = []; 
T = Y';
end                
            
        













