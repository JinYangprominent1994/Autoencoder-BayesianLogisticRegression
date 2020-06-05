A = importfile('iris.txt',52,151);

train1 = A(1:30,1:4);
test1 = A(31:50,1:4);
train2 = A(51:80,1:4);
test2 = A(81:100,1:4);

constant1 = ones(30,1);
constant2 = ones(20,1);

x_1 = table2array(train1);
x_2 = table2array(test1);
x_3 = table2array(train2);
x_4 = table2array(test2);

x_train1 = [constant1';x_1']';
x_test1 = [constant2';x_2']';
x_train2 = [constant1';x_3']';
x_test2 = [constant2';x_4']';

beta = zeros(1,5);
N_train = 30;
N_total_train = 60;
N_test = 20;

learning_rate = 0.01;
max_iterations = 100;
var = 1000;

% Initialize gradient with zero
gradient_temp = zeros(1,5);

energy = zeros(1,100);

% Implement gradient descent through iterations
for i = 1:max_iterations
    
    % Calculate gradient
    for j = 1:N_train    
        gradient_temp = gradient_temp + (norm(beta)./var - x_train1(j,:)'./ (exp(x_train1(j,:) * beta') + 1))';
    end
    
    for j = 1:N_train    
        gradient_temp = gradient_temp + (x_train2(j,:)'- x_train2(j,:)'./ (exp(x_train2(j,:) * beta') + 1) + norm(beta)./var)';
    end
    % Scale gradient with the number of training samples
    gradient = gradient_temp/N_total_train;
    
    % Based on calculated gradient to update beta
    beta = beta - learning_rate * gradient; 
    
    for j = 1:N_train    
       energy(i) = energy(i) + log(1 + exp(- x_train1(j,:) * beta')) + norm(beta).^2./(var*2);
       energy(i) = energy(i) + x_train2(j,:)* beta' + log(1 + exp(- x_train2(j,:) * beta')) + norm(beta).^2./(var*2); 
    end
end

% Initialize the classification error with zero
total_error_class = 0;
result1 = 0;
result2 = 0;

% Count the number of misclassified labels through iterations
for i = 1:N_test
    result1 =  x_test1(i,:) * beta';
    
    if 1/(1+ exp(-result1)) < 0.5
        total_error_class = total_error_class + 1;
    end
    
    result2 = x_test2(i,:) * beta';
    
    if 1/(1+ exp(-result2)) > 0.5
        total_error_class = total_error_class + 1;
    end
end

plot(energy);

error_rate = total_error_class / (N_test * 2);