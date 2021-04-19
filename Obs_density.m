function weigths=Obs_density(Ypoint,X,U,thetas)
% this function is model specific
% G( y_t | x_t ) ~ N(beta*x_t, exp(x_t/2))

w=Ypoint-thetas(4)*X-exp(X/2).*U*thetas(5);
sigma=exp(X/2)*sqrt(1-thetas(5)^2);
weigths=log(1./((2*pi)^(0.5).*sigma).*exp(-0.5*(w).^2./sigma.^2));
end