% Name - Yatharth Sharma
% knn clssifier  inorder to train the classifier uncomment the below section
% NOTE - load val with load the .mat file which cintains 101 nearest points


% fileLabelTrain = 'C:/Users/yatharth/Documents/project2/project2/knn/train-labels.idx1-ubyte'
% fileLabelTest = 'C:/Users/yatharth/Documents/project2/project2/knn/t10k-labels.idx1-ubyte'
% fileTrain = 'C:/Users/yatharth/Documents/project2/project2/knn/train-images.idx3-ubyte'
% fileTest = 'C:/Users/yatharth/Documents/project2/project2/knn/t10k-images.idx3-ubyte'
% y_train = loadMNISTLabels(fileLabelTrain);
% y_test =loadMNISTLabels(fileLabelTest);
% x_train = loadMNISTImages(fileTrain);
% x_test =loadMNISTImages(fileTest);
%
% d = pdist2(x_test',x_train');
%
% [~, ind] = sort(d,2);
% Fkneighbours = ind(:,1:1+100);
% index_size = size(Fkneighbours);
% index_size

% val = zeros(10000,101);
% for (i = 1: index_size(1,1))
%     for j =1: index_size(1,2)
%         
%         val(i,j) = y_train(Fkneighbours(i,j),1);
%     end
%     
% end

load val
valt = val';

 ks = [1,3,5,10,50,70,80,90,100]
 
 acc_final = zeros(1,9)
 i =1;
 for k = ks
     k;
     l = mode(valt(1:k,:),1);
     acc = l - y_test';
     acc_final(1,i) =sum(acc == 0)/100;
     i = i+1;
     acc_final;
 end
 save acc_final
figure
 plot(ks,acc_final)
title('KNN - accuracy vs k')
xlabel('K') % x-axis label
ylabel('accuracy (%) ') % y-axis label






