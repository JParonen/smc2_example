function [newstate,newZ,newX,newXweights,acc] = PMHkernel(oldZ,oldstate,oldX,oldXweights,ydata,Nsteps,R,N,Priors,acc)

prior_pdf_mu=@(theta) pdf(Priors.mu,theta);
prior_pdf_rho=@(theta) pdf(Priors.rho,theta);
prior_pdf_sigma=@(theta)  pdf(Priors.sigma,theta);
prior_pdf_beta=@(theta) pdf(Priors.beta,theta);
prior_pdf_phi=@(theta) pdf(Priors.phi,theta);
%%%%%
npar=length(oldstate);
for moves=1:Nsteps 
    proposal=oldstate+R*randn(npar,1);
    %%%%
    [Z_proposal,X_proposal,Xweights_proposal,errorflag]=PF_call(ydata,proposal,N);
    %%%%%%%%%%%%%%%%%%%%%%%%%
    if errorflag==0% saves time, look explanation inside PF_call
        acceptance_ratio=-Inf;
    else
        acceptance_ratio=(Z_proposal-oldZ)+...
            log(prior_pdf_mu(proposal(1))/prior_pdf_mu(oldstate(1)))+...
            log(prior_pdf_rho(proposal(2))/prior_pdf_rho(oldstate(2)))+...
            log(prior_pdf_sigma(proposal(3))/prior_pdf_sigma(oldstate(3)))+...
            log(prior_pdf_beta(proposal(4))/prior_pdf_beta(oldstate(4)))+...
            log(prior_pdf_phi(proposal(5))/prior_pdf_phi(oldstate(5)));
    end
    if log(rand()) <= min(1,acceptance_ratio) && isfinite(acceptance_ratio)
        oldstate=proposal;       
        % now since we accepted the new theta^{m}
        %its cumulative log-weigth changes as well.
        oldZ=Z_proposal;      
        %set the new X-particles (for particular theta^{m}) and their weigths because
        %the move was accepted so the states must move, too.
        oldX=X_proposal;
        oldXweights=Xweights_proposal;
        acc=acc+1;
    end
end
newZ=oldZ;
newstate=oldstate;
newX=oldX;
newXweights=oldXweights;
end
