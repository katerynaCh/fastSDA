function W = fastSDA(X_train, y_train, dim, alpha)
% X_train_sorted = train data, NxD
% y_train = train labels, Nx1
% dims = number of dimensions to keep 
% alpha = regularization parameter (0.001, 0.01, 0.1, 1, 10, 100, 1000 etc - the best is to select on validation set)
C = length(unique(y_train));
n_clusters = ceil((dim+1) / C);

%sort data by class
[y_train, sortIndex] = sort(y_train); 
X_train = X_train(sortIndex,:);

%mean-center
sampleMean = mean(X_train);
X_train = (X_train - repmat(sampleMean,size(X_train,1),1));

X_train_sorted = zeros(size(X_train));
clst_lbls = ones(size(y_train));
clst_class_lbls = n_clusters * ones(1,C);
idx = 1;
for i=1:C
  X_curr_class = X_train(y_train == i,:);
  if n_clusters == 1
      X_idx = ones(size(X_curr_class,1),1);
  else
      X_idx = fkmeans(X_curr_class,n_clusters);
  end
  for j=1:n_clusters
      this_clust = find(X_idx == j);
      X_curr_clust = X_curr_class(this_clust,:);   
      X_train_sorted(idx:idx+length(this_clust)-1,:)= X_curr_clust;
      clst_lbls(idx:idx+length(this_clust)-1) = j*ones(size(X_curr_clust,1),1);
      idx = idx+length(this_clust);
  end      
end

clear X_train;


X_train_sorted = X_train_sorted';

if size(X_train_sorted,1) < size(X_train_sorted,2)
    
    ss = X_train_sorted*X_train_sorted';
    ss = ss + alpha*eye(size(ss));
    
    [R,p] = chol(ss);
    T = calculate_targets_singleview(y_train, clst_class_lbls', clst_lbls);
    
    W = R\(R'\(X_train_sorted*T'));
    
    %W = pinv(X_train_sorted*X_train_sorted')*X_train_sorted*T';
else
    ss = X_train_sorted'*X_train_sorted;
    ss = ss + alpha*eye(size(ss));
    
    [R,p] = chol(ss);
    T = calculate_targets_singleview(y_train, clst_class_lbls', clst_lbls);
    
    W = X_train_sorted*(R\(R'\(T')));
    %W = pinv(X_train_sorted*X_train_sorted')*X_train_sorted*T';
end

W = W(:,1:dim);

%orthogonalize projection matrix
W = orth(W);
