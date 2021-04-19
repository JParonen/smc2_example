%% Nx and acc. rates

figure('name','Nx')
nn=size(NX,2);
jitter=20*rand(nn,1);
for i=1:nn
    stairs(NX(:,i)+jitter(i),'Linewidth',1.4);hold on
end
xlabel('Time','FontSize',14,'Interpreter','latex');
ylabel('$N_x$','Interpreter','latex','FontSize',14);grid on
title('Adding particles','FontSize',14,'Interpreter','latex')
ylim([0 Np_x_max])
figure('name','Acceptance rates');
for kk=1:nn
    id=find(accept_rates(:,2,kk)>0);
    ac=accept_rates(id,:,kk);
    h=plot(ac(:,1),ac(:,2),':s','MarkerSize',3);hold on;
    set(h, 'MarkerFaceColor', get(h,'Color'));
end
plot([0 T],[acc_threshold acc_threshold],'k--','Linewidth',1);axis tight
xlabel('Time','FontSize',14,'Interpreter','latex');
ylabel('Accept rates','FontSize',14,'Interpreter','latex');

%% kernel densities: rho

colors={[0, 0.4470, 0.7410],...
    [0.8500, 0.3250, 0.0980],...
    [0.9290, 0.6940, 0.1250],...
    	[0.4940, 0.1840, 0.5560],...
         	[0.4660, 0.6740, 0.1880],...
            [0.3010, 0.7450, 0.9330],...
            [0.6350, 0.0780, 0.1840],...
            	[0, 0.75, 0.75],...
                	[0.4660, 0.6740, 0.1880],...
                    [0.25 0.25 0.25]};
clear ff
clear xx
figure('name','density_rho')
temp=[1:nruns];
jj=1;
for j=temp
    [ff(jj,:),xx(jj,:)]=ksdensity(thetaParticles(2,:,T,j)); 
    jj=jj+1;
end
jj=1;
for jj=1:length(temp)
    fill(xx(jj,:),ff(jj,:),colors{jj},'FaceAlpha',0.2)
    hold on
end
plot([params.rho params.rho],[0 max(max(ff))],'k--','Linewidth',1.5)
xlabel('$\rho$','FontSize',20,'Interpreter','latex');
title(sprintf('$t = %d$',T),'FontSize',18,'Interpreter','latex')
axis tight
%% kernel densities: mu

clear ff
clear xx
figure('name','density_mu')
temp=[1:nruns];
jj=1;
for j=temp
    [ff(jj,:),xx(jj,:)]=ksdensity(thetaParticles(1,:,T,j)); 
    jj=jj+1;
end
jj=1;
for jj=1:length(temp)
    fill(xx(jj,:),ff(jj,:),colors{jj},'FaceAlpha',0.2)
    hold on
end
plot([params.mu params.mu],[0 max(max(ff))],'k--','Linewidth',1.5)
xlabel('$\mu$','FontSize',20,'Interpreter','latex');
title(sprintf('$t = %d$',T),'FontSize',18,'Interpreter','latex')
axis tight

%% kernel densities: sigma

clear ff
clear xx
figure('name','density_sigma')
temp=[1:nruns];
jj=1;
for j=temp
    [ff(jj,:),xx(jj,:)]=ksdensity(thetaParticles(3,:,T,j)); 
    jj=jj+1;
end
jj=1;
for jj=1:length(temp)
    fill(xx(jj,:),ff(jj,:),colors{jj},'FaceAlpha',0.2)
    hold on
end
plot([params.sigma params.sigma],[0 max(max(ff))],'k--','Linewidth',1.5)
xlabel('$\sigma$','FontSize',20,'Interpreter','latex');
title(sprintf('$t = %d$',T),'FontSize',18,'Interpreter','latex')
axis tight

%% kernel densities: beta (skewness)

clear ff
clear xx
figure('name','density_beta')
temp=[1:nruns];
jj=1;
for j=temp
    [ff(jj,:),xx(jj,:)]=ksdensity(thetaParticles(4,:,T,j)); 
    jj=jj+1;
end
jj=1;
for jj=1:length(temp)
    fill(xx(jj,:),ff(jj,:),colors{jj},'FaceAlpha',0.2)
    hold on
end
plot([params.beta params.beta],[0 max(max(ff))],'k--','Linewidth',1.5)
xlabel('$\beta$','FontSize',20,'Interpreter','latex');
title(sprintf('$t = %d$',T),'FontSize',18,'Interpreter','latex')
axis tight

%% kernel densities: phi (leverage)

clear ff
clear xx
figure('name','density_phi')
temp=[1:nruns];
jj=1;
for j=temp
    [ff(jj,:),xx(jj,:)]=ksdensity(thetaParticles(5,:,T,j)); 
    jj=jj+1;
end
jj=1;
for jj=1:length(temp)
    fill(xx(jj,:),ff(jj,:),colors{jj},'FaceAlpha',0.2)
    hold on
end
plot([params.phi params.phi],[0 max(max(ff))],'k--','Linewidth',1.5)
xlabel('$\phi$','FontSize',20,'Interpreter','latex');
title(sprintf('$t = %d$',T),'FontSize',18,'Interpreter','latex')
axis tight

%% plot variance of marginal likelihood

tem=Z;
tem(~isfinite(tem))=NaN;
figure('name','marginal log-likelihoog');
for i=size(tem,3)
    va=std(squeeze(Z(:,:,i)),[],1,'omitnan');
    plot(va);hold on
end
xlabel('Time');
ylabel('std of marginal log-lik')
title('Std of $\log(p_{\theta}(y_{1:t}))$','Interpreter','latex')