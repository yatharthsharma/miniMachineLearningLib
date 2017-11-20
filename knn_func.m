function [accuracy]  = knn_func(x_train,y_train,x_test,y_test,k)
    d = pdist2(x_test,x_train);
    [~, ind] = sort(d,2);
     Fkneighbours = ind(:,1:1+k);
index_size = size(Fkneighbours);
index_size
val = zeros(size(y_test,1),k+1);
for i = 1: index_size(1,1)
    for j =1: index_size(1,2)
        
        val(i,j) = y_train(Fkneighbours(i,j),1);
    end
end
    valt = val';
    
    pred = mode(valt(1:k,:),1);
     
     acc = pred' - y_test;
     accuracy =(sum(acc == 0)/1598)*100;
   
    
end