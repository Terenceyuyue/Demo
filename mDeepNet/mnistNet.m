function [f,gw2,gw3,gb2,gb3] = mnistNet(mini_batch_x, mini_batch_y, w2,w3,b2,b3)

sigmoid = @(z) 1./(1+exp(-z));

% forward
a1 = mini_batch_x;
z2 = w2*a1 + b2;
a2 = sigmoid(z2);
z3 = w3*a2 + b3;
a3 = sigmoid(z3);

% loss value
f = sum((mini_batch_y(:) - a3(:)).^2);

% gradient values of free parameters
[gw2,gw3,gb2,gb3] = dlgradient(f,w2,w3,b2,b3);

