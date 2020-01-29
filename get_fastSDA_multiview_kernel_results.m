function [accuracy, time] = get_fastSDA_multiview_kernel_results(Ktrain_proto, Ktest_proto, Kref_proto, y_train, y_test, clst_class_lbls, cluster_labels, alpha, knn_n)

    % Ktrain_proto =  Map with key = 'v', value = kernel matrix of train data
    % Ktest_proto =  Map with key = 'v', value = kernel matrix of test data
    % Kref_proto =  Map with key = 'v', value = kernel matrix of reference
    % data (train data in full kernel case)
    % Lval = number of reference vectors in the reduced kernel case, N in
    % full kernel case
    % y_train = train labels (size Nx1)
    % y_test = test_labels (size Nx1)
    % clst_class_lbls = number of clusters in each class, size 1xC
    % cluster labels = Map with key = 'v', value = cluster labels in view v
    % alpha = regularization parameter
    % knn_n = number of neighbors to be used in kNN classifier
    
    V = length(Ktrain_proto);
    N = length(y_train);
    W = containers.Map;
    X_train_total = [];
    X_test_total = [];
    Lval = size(Ktrain_proto('1'),1)
    constr = tic;
    

    Ktrain_diag = zeros(Lval*V, N*V);
    L = size(Ktrain_diag('1'),1)*V;
    for v = 1:V
        Ktrain_diag(v*Lval - Lval + 1: v*Lval,v*N - N + 1: v*N) = Ktrain_proto(int2str(v));
    end
         
    time_constr = toc(constr);
    W = containers.Map;
    X_train_transformed = containers.Map;
    X_test_transformed = containers.Map;
    X_train_total = [];
    X_test_total = [];
    time_our1 = tic;

    Tgen = calculate_target_vectors_multiview(y_train,clst_class_lbls, cluster_labels)';
    Ws = [];
    L = size(Ktrain_proto('1'),1)*V;
    Ktrain2=Ktrain_diag*Ktrain_diag'+alpha*eye(L);   

    [R,p]=chol(Ktrain2);
    while p~=0
        Ktrain2 = Ktrain2 + alpha*eye(L);
        [R,p] = chol(Ktrain2);
    end 
	A_tot = R\(R'\(Ktrain_diag*Tgen));
    for view = 1:V
        A = A_tot(Lval*view - Lval + 1:Lval*view,:);
        tmpNorm = sqrt(diag(A'*Kref_proto(int2str(view))*A));
        A = A./repmat(tmpNorm',size(A,1),1);
        W(int2str(view)) = A;
        X_temp = W(int2str(view))'*Ktrain_proto(int2str(view));
        X_train_total = [X_train_total; X_temp];
    end
	time_kernel = toc(time_our1); time = time_kernel+time_constr;
	for view = 1:V
        X_test_temp = W(int2str(view))'*Ktest_proto(int2str(view));
        X_test_total = [X_test_total; X_test_temp];
    end
	clear Tgen; clear W; clear X_test_transformed; clear X_train_transformed;
	Mdl = fitcknn(X_train_total',y_train,'NumNeighbors',knn_n);
	predictions = Mdl.predict(X_test_total');
	accuracy = sum(y_test == predictions) / numel(y_test); 
	clear X_test_total_our; clear X_train_total_our;
end