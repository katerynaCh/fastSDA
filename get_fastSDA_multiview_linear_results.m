function [accuracy, time] = get_fastSDA_multiview_linear_results(X_train, X_test, y_train, y_test, Ds, clst_class_lbls, cluster_labels, alpha, knn_n)
    % X_train = Map with key = 'v', value = train data of view v, size Nxd
    % X_test = Map with key = 'v', value = test data of view v, size Nxd
    % Ds = Map{'v' -> dimension of view v}
    % y_train = train labels (size Nx1)
    % y_test = test_labels (size Nx1)
    % clst_class_lbls = number of clusters in each class, size 1xC
    % cluster labels = Map with key = 'v', value = cluster labels in view v
    % alpha = regularization parameter
    % knn_n = number of neighbors to be used in kNN classifier
    
    V = length(X_train);
    N = size(X_train('1'),1);
    W = containers.Map;
    X_train_total = [];
    X_test_total = [];
    sum_dim = 0;
    for v = 1:V
        sum_dim = sum_dim + size(X_train(int2str(v)), 2);
    end
    X_diag = zeros(sum_dim, N*V);
    start = 0;
    for v = 1:V
        if v ~= 1
            start =  start + Ds(int2str(v-1));
        end
            X_diag(start + 1: start + Ds(int2str(v)),v*N-N+1:v*N) = X_train(int2str(v))';
    end

    time_start = tic;

    Tgen = calculate_target_vectors_multiview(y_train, clst_class_lbls, cluster_labels)'; 
    X_diag2 = X_diag*X_diag'+alpha*eye(size(X_diag,1));
    [R,p]=chol(X_diag2);
    L = size(X_diag2,1);
    while p~=0
        X_diag2 = X_diag2 + alpha*eye(L);
        [R,p] = chol(X_diag2);
    end 
    W_r = R\(R'\(X_diag*Tgen));

    start = 0;
    for view = 1:length(X_train)
        if view ~= 1
           start =  start + Ds(int2str(view-1));
        end

        W(int2str(view)) = W_r(start + 1:start + Ds(int2str(view)),:);
        tmpNorm = sqrt(diag( W(int2str(view))'* W(int2str(view)))); %normalize each W
        W(int2str(view)) =  W(int2str(view))./repmat(tmpNorm',size( W(int2str(view)),1),1); 
        X_temp = W(int2str(view))'*X_train(int2str(view))';
        X_train_total = [X_train_total; X_temp];
    end
    time = toc(time_start);
    
    for view = 1:length(X_train)
        X_test_temp = W(int2str(view))'*X_test(int2str(view))';
        X_test_total = [X_test_total; X_test_temp];
    end
    
    clear Tgen; clear W; clear X_test_transformed; clear X_train_transformed;
    Mdl = fitcknn(X_train_total',y_train,'NumNeighbors',knn_n);
    predictions = Mdl.predict(X_test_total');
    accuracy = sum(y_test == predictions) / numel(y_test); 
    clear X_test_total_our; clear X_train_total_our;
end