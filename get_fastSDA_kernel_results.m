function [accuracies, times] = get_fastSDA_kernel_results(Ktrain, Ktest, Kref, y_train, y_test, clst_class_lbls, clst_lbls, knn_ns, ReguAlpha)
%Ktrain = NxN kernel matrix of training data
%Ktest = kernel matrix of validation data
%Kref = kernel matrix of reference data to be used for approximate regression (=Ktrain for normal case)
%y_train = 1xN  class labels of train data
%y_test = 1xN class labels of validation data
%clst_class_lbls = 1xC vector with number of clusters in each class
%clst_lbls = 1xN cluster labels
%knn_ns = numbber of neighbors for kNN classification (e.g. [1,3,5,7])
%ReguAlpha = regularization parameter for inversion 
times = [];
accuracies = [];
n_clusters = length(unique(clst_lbls));
C = length(unique(y_train));
start = tic;
T = calculate_targets_singleview(y_train, clst_class_lbls', clst_lbls);

L = size(Ktrain,1);
Ktrain2=Ktrain*Ktrain'+ReguAlpha*eye(L);   
[R,p]=chol(Ktrain2);
while p~=0
    Ktrain2 = Ktrain2 + ReguAlpha*eye(L);
    [R,p] = chol(Ktrain2);
end 
A = R\(R'\(Ktrain*T'));
 
tmpNorm = sqrt(diag(A'*Kref*A));
A = A./repmat(tmpNorm',size(A,1),1);

time = toc(start);
X_train_tr = A'*Ktrain;  
X_test_tr = A'*Ktest;

for knn_n = knn_ns
    Mdl = fitcknn(X_train_tr',y_train,'NumNeighbors',knn_n);
    predictions = Mdl.predict(X_test_tr');
    accuracy = sum(y_test == predictions) / numel(y_test);
    accuracies = [accuracies accuracy];
    times = [times time];
end
end
