function [y,x]=simulateData(params,numObs)
%%%%%
mu=params.mu;
rho=params.rho;
sigma=params.sigma;
beta=params.beta;
phi=params.phi;
%%%%%
u=randn();
v=phi*u+sqrt(1-phi^2)*randn();
x(1)=mu+sqrt(sigma^2/(1-rho^2))*u; %initial value
y(1)=beta*x(1)+exp(x(1)/2)*v;
for t=2:numObs
    U=randn();
    V=phi*U+sqrt(1-phi^2)*randn();
    x(t)=mu+rho*(x(t-1)-mu)+sigma*U;
    y(t)=beta*x(t)+exp(x(t)/2)*V;
end
end