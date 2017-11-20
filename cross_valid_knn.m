function[accuracy] = cross_valid_knn(x_train,y_train,x_test,y_test,k)
size_train = size(x_train);
ix = randperm(size(x_train,1));

x_train = x_train(ix,:);
y_train = y_train(ix,:);
size_train = size(x_train);
count_k_fold = 0;
accAvg = 0;
for it = 1:5
       k_fold_x_test = x_train(count_k_fold+1:it*1000,:);
       k_fold_x_train = vertcat(x_train(1:count_k_fold,:),x_train(it*1000+1:5000,:));
       
       k_fold_y_test = y_train(count_k_fold+1:it*1000,:);
       k_fold_y_train = vertcat(y_train(1:count_k_fold,:),y_train(it*1000+1:5000,:));
       
       d = pdist2(k_fold_x_test,k_fold_x_train);
       [~, ind] = sort(d,2);
       Fkneighbours = ind(:,1:k);
       index_size = size(Fkneighbours);
        val = zeros(1000,k);
        for i = 1: index_size(1,1)
            for j =1: index_size(1,2)
                val(i,j) = k_fold_y_train(Fkneighbours(i,j),1);
            end
            
        end      
       
    valt = val';
    pred = mode(valt(1:k,:),1);
    acc = pred' - k_fold_y_test;
   accAvg = accAvg + sum(acc == 0)/10;
    
    count_k_fold = it*1000;
    
end

accuracy = accAvg/5;

end
