function [newZ, newXparts, newXweights,flag]=PF_call(Yt,theta,Nx,add)

% BOOTSTRAP PARTICLE FILTER
%this function computes p_bar (y_{1:t}) for each individual theta-particle
%
%INPUT:  Yt:  Observed data until time t (NOTE: not the "T" as in the main)
%        theta: parameters (5x1 vector in this case)
%        Nx: number of X-particles
%        add: 0 or 1, 1 if particle addition happens
         
% OUTPUT: newW: likelihood of proposal theta
%         newXparts: propagated X-particles at time T
%         newXweights: weigths of those particles at time T
%         flag:  Error/exit flag in case weigths are really close to zero
if ~exist('add','var')
    add=0;
end
T = length(Yt);
X_weights = zeros(T,Nx); %weigths
Xparticles = zeros(T,Nx);%particles
% first state
Xparticles(1,:)=theta(1)+sqrt(theta(3)^2/(1-theta(2)^2))*randn(1,Nx);
flag=1;
u=randn(1,Nx);
for t = 1:T
    if t==1
        X_weights(t,:)=Obs_density(Yt(t),Xparticles(t,:),u,theta); %save the weigths
    else
        a=Resample(weight_mod,Nx);  %resample every time
        [Xparticles(t,:),u]=X_transition(Xparticles(t-1,a),theta,Nx);%propagate forward
        X_weights(t,:)=Obs_density(Yt(t),Xparticles(t,:),u,theta); %save the weigths
    end
    weight_mod=exp(X_weights(t,:)-max(X_weights(t,:)));
    if (sum(weight_mod)<1e-15 || ~isreal(weight_mod))
        %if weigths "carry no meaning" == really small, discard
        %them and this proposal theta
        flag=0;
        break;
    end
    weight_mod = weight_mod/sum(weight_mod); %normalize
end
if flag==0
    newZ=[];
    newXparts=[];
    newXweights=[];
    return;
end
if add
    newXparts = Xparticles; %save the last x-particles
    newXweights = X_weights;% and their weigths
    %compute the p(y_{1_t}): sum all the log-likelihood increments
    newZ = cumsum(log(1/Nx*sum(exp(X_weights),2)));
else
    newXparts = Xparticles(T,:); %save the last x-particles
    newXweights = X_weights(T,:);% and their weigths
    %compute the p(y_{1_t}): sum all the log-likelihood increments
    newZ = sum(log(1/Nx*sum(exp(X_weights),2)));
end

end