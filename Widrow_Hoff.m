function [test_targets] = ...
    Widrow_Hoff(train_patterns, train_targets, test_patterns)
[c, n]		   = size(train_patterns);
Max_iter=1000;b_min=0.1;eta= 0.01;

train_patterns  = [train_patterns , ones(c,1)];
train_zero      = find(train_targets == 0);
processed_patterns = train_patterns;
processed_patterns(train_zero,:) = ...
    -processed_patterns(train_zero,:);

b                  = ones(1,c);
Y                  = processed_patterns;
a                  = rand(n+1,1);
k	               = 0;
e    	           = 1e3;

while ((sum(abs(e) > b_min)>0) && (k < Max_iter))
    k = k+1;
    e       = (Y * a)' - b;
    for i=1:c;
        a = a - eta*Y(i,:)'*(Y(i,:)*a-b(i));
    end
end
test_targets =...
    (a' * [test_patterns, ones(size(test_patterns,1),1)]')';
if (length(unique(train_targets)) == 2)
    test_targets = test_targets > 0;
end
end