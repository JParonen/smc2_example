function [Xnew,U]=X_transition(Xold,theta,Nx)
% this function is model specific
% Markov transition of the model f(x_{t} | x_{t-1})
U=randn(1,Nx);
Xnew=theta(1)+theta(2).*(Xold-theta(1))+theta(3).*U;
end