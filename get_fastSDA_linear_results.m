function [times, accuracies] = get_fastSDA_linear_results(X_train_sorted, X_test, y_train, y_test, clst_class_lbls,clst_lbls,knn_ns,alpha)
%X_train_sorted = clustered train data, sorted according to clusters and classes, i.e. [c1s1, c1s1, c1s2, c1s2, c2s1, c2s1, c2s2, ...]
%X_test = test data
%y_train = trainl labels, sorted
%y_test = test labels
%clst_class_lbls = amount of clusters in each class
%clst_lbls = cluster label for each training instance in X_train_sorted
%knn_ns = number of neighbours to use in kNN classification (e.g. [1,3,5,7])

times = [];
accuracies = [];
X_train_sorted = X_train_sorted';
if size(X_train_sorted,1) < size(X_train_sorted,2)
    
    ss = X_train_sorted*X_train_sorted';
    [R,p] = chol(ss);
    while p~=0
        ss = ss + alpha*eye(size(ss)); 
        [R,p] = chol(ss);
    end
    start_time = tic;
    T = calculate_targets_singleview(y_train, clst_class_lbls', clst_lbls);
    dd = X_train_sorted*X_train_sorted';
    [R,p] = chol(ss);
    W = R\(R'\(X_train_sorted*T'));
else
    ss = X_train_sorted'*X_train_sorted;
    [R,p] = chol(ss);
    while p~=0
        ss = ss + alpha*eye(size(ss));
        [R,p] = chol(ss);
    end
    start_time = tic;
    T = calculate_targets_singleview(y_train, clst_class_lbls', clst_lbls);
    dd = X_train_sorted'*X_train_sorted;
    [R,p] = chol(ss);
    W = X_train_sorted*R\(R'\(T'));
end

tmpNorm = sqrt(diag(W'*W));
W = W./repmat(tmpNorm',size(W,1),1);

time = toc(start_time);

X_train_sorted =X_train_sorted';

X_train_transformed = W'*X_train_sorted';
X_test_transformed = W'*X_test';

for knn_n = knn_ns
    Mdl = fitcknn(X_train_transformed',y_train,'NumNeighbors',knn_n);
    predictions = Mdl.predict(X_test_transformed');
    accuracy = sum(y_test == predictions) / numel(y_test);
    accuracies = [accuracies accuracy];
    times = [times time];
end

