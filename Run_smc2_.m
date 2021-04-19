%  
%  This script performs parameter estimation with smc2 without using parallel computing
%  (Code can be easily modified to do each individual run of smc2 in
%  parallel)

% This code implements smc2 with a state-process that is one-dimensional.
% However, with slight modifications, multidimensional state-process can be
% used as well

% If one wants to use this script with another model, atleast functions
% "X_transition" ans "Obs_density must be modified along with small details
% in "PF_call" and "PMHkernel" that relate to number of parameters, initial
% X and priors.

% This code was made for Master's thesis: "Sequential estimation of
% state-space model parameters" by Joona Paronen
% see the thesis for references
%% simulate data

clc;clear all;close all

%example simple stochastic volatility model: 
%-------------------------------------------------------
%x_0 ~ N(mu, sigma^2/(1-rho^2))
%x_t = mu + rho*(x_{t-1}-mu) + sigma*U_t,  U ~ N(0,1)
%y_t|x_t =beta*x_t+exp(x_t/2)*V_t
%corr(U,V)=phi
%<<<<
%parameter vector: THETA=(mu, rho, sigma, beta, phi)

% define prior densities:
%-------------------------------------------------------
Priors.mu=makedist('Normal',0,2);
Priors.rho=makedist('Beta',9,1);
Priors.sigma=makedist('Gamma',2,2);
Priors.beta=makedist('Normal',0,1);
Priors.phi=makedist('Uniform',-1,1);
%-------------------------------------------------------
params.mu=-1;
params.rho=0.87;
params.sigma=1.1;
params.beta=0;
params.phi=-0.44;
% params.mu=random(Priors.mu);
% params.rho=random(Priors.rho);
% params.sigma=random(Priors.sigma);
% params.beta=random(Priors.beta);
% params.phi=random(Priors.phi);

T=600; % number of observations
npar=length(fieldnames(params));% # of model parameters
[y,x]=simulateData(params,T);
subplot(211);
plot(x,'Color','k');
title('State process X_t');
subplot(212)
plot(y.^2,'Color','k');
title('Squared observations Y_t');xlabel('t');
y=y(:);x=x(:);
%% SMC^2 settings

% necessary inputs:
%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
NT=1000;                        % # of parameter particles
init_X_particles=100;           % # of initial x-particles
Np_x_max=1000;                  % maximum # of x-particles (for allocation, can be as big as memory allows)
ESSfrac=0.5;                    % effective sample size = NT * ESSfrac, choose between 0 and 1
Nsteps=3;                       % how many rejuvenation steps
acc_threshold=0.15;             % guideline to tune covariance and add x-particles 
nruns=1;                        % how many runs of smc2
%<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
% mcmc settings:
sd=2.38^2/npar; %scaling factor from literature
e=1e-10; %to prevent singularity
I=eye(npar);
%>>>>>>>>>>>>>>>>>>>>>>>>>>>

T=length(y); % # of observations
%sample initial theta-particles
thetaParticles=zeros(npar,NT,T+1,nruns); % allocate for each timestep
fld=fieldnames(Priors);
for i=1:npar
    thetaParticles(i,:,1,:)=random(Priors.(fld{i}),1,NT,1,nruns);
end
%allocate variables that are saved
NX = zeros(T,nruns);
accept_rates=zeros(T,2,nruns);
Z=zeros(NT,T,nruns);

%% Run 'nruns' of smc2
for run=1:nruns
    NX(:,run)=init_X_particles;
    
    % These can be saved for each run if one is interested to do filtering/smoothing
    % or plotting ESS
    %------------------------------
    Xweights=zeros(T,Np_x_max,NT);  
    Xparticles=zeros(T,Np_x_max,NT); 
    ParamWeights=zeros(NT,T); 
    ESS=zeros(T,1);
    %------------------------------
    % Particle system is a set of variables:
    % ( thetaParticles, Xparticles, Xweights, Paramweights )
    U=randn(1,NX(1,run));
    c=1;R=1;
    for t=1:T % go through every timestep until T      
        %Move one step ahead with particle filter
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for m = 1:NT %for each theta-particle (m)
            if t==1
                Xweights(t,1:NX(t,run),m)=Obs_density(y(t),Xparticles(t,1:NX(t,run),m),U,thetaParticles(:,m,t,run));
            else
                % practical trick to maintain the right order of weights
                % while making them larger (effect is eliminated in
                % normalization)
                temp=exp(Xweights(t-1,1:NX(t-1,run),m)-max(Xweights(t-1,1:NX(t-1,run),m)));
                temp=temp/sum(temp);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%
                a=Resample(temp,NX(t,run)); %resample
                %propagate through system model
                [Xparticles(t,1:NX(t,run),m),U]=X_transition(Xparticles(t-1,a,m),thetaParticles(:,m,t,run),NX(t,run));
                Xweights(t,1:NX(t,run),m)=Obs_density(y(t),Xparticles(t,1:NX(t,run),m),U,thetaParticles(:,m,t,run));
            end              
        end
        % Compute incremental likelihood and update marginal log-likelihood
        % and parameter weights
        %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        % sum over x-particles to get weight of each theta-particle
        if t==1
            ParamWeights(:,1)=squeeze(sum(1/NX(t,run)*exp(Xweights(t,1:NX(t,run),:)),2));
            Z(:,1,run)=log(ParamWeights(:,1));
        else
            likelihood_increment=squeeze(sum(1/NX(t,run)*exp(Xweights(t,1:NX(t,run),:)),2));         
            ParamWeights(:,t)=ParamWeights(:,t-1).*likelihood_increment;
            
            % marginal likelihood is the product of incremental likelihood
            Z(:,t,run)=Z(:,t-1,run)+log(likelihood_increment);
        end
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        % normalize
        ParamWeights(:,t)=ParamWeights(:,t)./sum(ParamWeights(:,t));
        ESS(t)=1/sum(ParamWeights(:,t).^2);       
        do_we_move=0; %dummy to indicate whether move is done at time t
        acc=0;%helper variable (temporary)
        if ESS(t)/NT<ESSfrac || isnan(ESS(t)) || ~isreal(ESS(t))
            do_we_move=1; %yes, we do move
            %resample theta-weigths (most meaningful parameters weigths)
            A=Resample(ParamWeights(:,t),NT);
            
            % Set the resampled particle system 
            %---------------------------------------------------
            thetaParticles(:,:,t,run)=thetaParticles(:,A,t,run);
            Z(:,t,run)=Z(A,t,run);
            Xparticles(t,1:NX(t,run),:)=Xparticles(t,1:NX(t,run),A);            
            Xweights(t,1:NX(t,run),:)=Xweights(t,1:NX(t,run),A);
            %---------------------------------------------------
            % Adaptation/tuning of proposal
            covariance=cov(thetaParticles(:,:,t,run)'); 
            if norm(covariance)<1e-12
                warning('Covariance too low, consider stopping')
                R=1e-2*I;
                c=1;
            else
                R=chol(c*sd*covariance+e*I,'lower');%cholesky decomposition
            end
            fprintf('Rejuvenating particles.\n');
            for m=1:NT  %
                [newpars,newZ,newXs,newXweights,acc]=PMHkernel(Z(m,t,run),...
                    thetaParticles(:,m,t,run),...
                    Xparticles(t,1:NX(t,run),m),...
                    Xweights(t,1:NX(t,run),m),...
                    y(1:t),...
                    Nsteps,R,NX(t,run),Priors,acc);
                
                % Set new particle system for m
                %-----------------------------------------
                thetaParticles(:,m,t+1,run)=newpars;
                Z(m,t,run)=newZ;
                Xparticles(t,1:NX(t,run),m)=newXs;
                Xweights(t,1:NX(t,run),m)=newXweights;
                %-----------------------------------------
            end
            % Set weights back to equal
            ParamWeights(:,t)=1;
        else
            % if rejuvenation was not done at time t
            thetaParticles(:,:,t+1,run)=thetaParticles(:,:,t,run);
        end
        % Checking the acc. rate
        accept_rates(t,:,run)=[t,acc/(Nsteps*NT)];
        if do_we_move
            if accept_rates(t,2,run)<acc_threshold
                fprintf('Acceptance rate below threshold %.2f\n',acc_threshold);
                c=c*2^(-5*(acc_threshold-accept_rates(t,2,run)));
                if NX(t,run)*1.5>Np_x_max
                    fprintf('Cannot add more x-particles.\n');
                else
                    fprintf('Adding more particles.\n')
                    NX(t+1:end,run)=floor(NX(t,run)*1.5);
                end               
            else
                fprintf('Acceptance rate %.4f\n',accept_rates(t,2,run));
            end
        end
        % print the progress and ESS and norm(R) for monitoring
        % 
        fprintf('RUN %d/%d, Iteration: %d/%d, ESS = %.2f, norm of cholcov = %.4f\n',run,nruns,t,T,ESS(t),norm(R));
    end
end

%% plots kernel densities in different time points

%choose:
whichRun=1;
%choose time points in descending order
TimePoints=[600 500 400];


colors={'k','c''y','m','b','r','g',};
i=1;
subplot(311)
co=[0,0,0];

for k=TimePoints
    if k==TimePoints(1)
        [FF,xx]=ksdensity(thetaParticles(1,:,k,whichRun));
    end
    [ff,xx]=ksdensity(thetaParticles(1,:,k,whichRun));
    text=sprintf('%d',k);
    plot(xx,ff,'DisplayName',text,'Color',co+0.2*i,'Linewidth',1.6-0.15*i);hold on
    i=i+1;
end
plot([params.mu params.mu],[0 max(FF)],'k:','Linewidth',2);
%legend show;
title(sprintf('\\mu = %.3f',params.mu))

i=1;
subplot(312)
for k=TimePoints
    if k==TimePoints(1)
        [FF,xx]=ksdensity(thetaParticles(2,:,k,whichRun));
    end
    [ff,xx]=ksdensity(thetaParticles(2,:,k,whichRun));
    text=sprintf('%d',k);
    plot(xx,ff,'DisplayName',text,'Color',co+0.2*i,'Linewidth',1.6-0.15*i);hold on
    i=i+1;
end
plot([params.rho params.rho],[0 max(FF)],'k:','Linewidth',2);
%legend show;
title(sprintf('\\rho = %.3f',params.rho))

i=1;
subplot(313)
for k=TimePoints
    if k==TimePoints(1)
        [FF,xx]=ksdensity(thetaParticles(3,:,k,whichRun));
    end
    [ff,xx]=ksdensity(thetaParticles(3,:,k,whichRun));
    text=sprintf('%d',k);
    plot(xx,ff,'DisplayName',text,'Color',co+0.2*i,'Linewidth',1.6-0.15*i);hold on
    i=i+1;
end
plot([params.sigma params.sigma],[0 max(FF)],'k:','Linewidth',2);
%legend show;
title(sprintf('\\sigma = %.3f',params.sigma))


figure
i=1;
subplot(211)
for k=TimePoints
    if k==TimePoints(1)
        [FF,xx]=ksdensity(thetaParticles(4,:,k,whichRun));
    end
    [ff,xx]=ksdensity(thetaParticles(4,:,k,whichRun));
    text=sprintf('%d',k);
    plot(xx,ff,'DisplayName',text,'Color',co+0.2*i,'Linewidth',1.6-0.15*i);hold on
    i=i+1;
end
plot([params.beta params.beta],[0 max(FF)],'k:','Linewidth',2);
%legend show;
title(sprintf('\\beta = %.3f',params.beta))
i=1;
subplot(212)
for k=TimePoints
    if k==TimePoints(1)
        [FF,xx]=ksdensity(thetaParticles(5,:,k,whichRun));
    end
    [ff,xx]=ksdensity(thetaParticles(5,:,k,whichRun));
    text=sprintf('%d',k);
    plot(xx,ff,'DisplayName',text,'Color',co+0.2*i,'Linewidth',1.6-0.15*i);hold on
    i=i+1;
end
plot([params.phi params.phi],[0 max(FF)],'k:','Linewidth',2);
%legend show;
title(sprintf('\\phi = %.3f',params.phi))