filName = 'knn_data.mat'
knn_data = load(filName)
% shuffledArray = orderedArray(randperm(size(orderedArray,1)),:);
y_train =  knn_data.train_label;
y_test = knn_data.test_label;

x_train = knn_data.train_data;
x_test =knn_data.test_data;
l = [0,1,2,3,4,5,6,7,8]
    
size_train = size(x_train);
x_train = x_train - mean(x_train(:));
x_train = x_train/std(x_train(:));

x_test= x_test - mean(x_test(:));
x_test = x_test/std(x_test(:));


cMat_train = cov(x_train');
cMat_test = cov(x_test');
[UTr,VTr,DTr] = svd(cMat_train);
[UTs,VTs,DTs] = svd(cMat_test);
VReduced = UTr(:,1:50);
XReduded = UTr*x_train;



VReducedTs = UTs(:,1:50);
XRedudedTs = UTs*x_test;

l = [0,1,2,3,4,5,6,7,8];
KPar  = 2*l + 1;
accuracy = zeros(1,9);
for it = 1:size(KPar,2)
    accuracy(1,it) = cross_valid_knn(XReduded,y_train,XRedudedTs,y_test,KPar(it));
end

figure
plot(KPar,accuracy);

knn_func(XReduded,y_train,XRedudedTs,y_test,11)



