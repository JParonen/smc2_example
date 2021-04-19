function [resampled_indices] = residual(w,N)

L=length(w);
resampled_indices=zeros(N,1);
% 
Nhat=floor(L.*w);%how many to include at least
if ~isreal(Nhat)
    resampled_indices=[1:N]';
    warning('Resampling failed');
    return;
end
sumNhat=sum(Nhat);
% remainder:
R=L-sumNhat;
% new weights for multinomial
Ws = (L.*w-floor(L.*w))/R;
i=1;
for j=1:L
    for k=1:Nhat(j)
        resampled_indices(i,1)=j;
        i = i+1;
    end
end
%multinomial:
cdf_weights = cumsum(Ws);
while i<=N
    r=rand();
    j=1;
    while (cdf_weights(j)<r)
        j=j+1;
        if j>length(cdf_weights)
            warning('Resampling failed!');
            resampled_indices=[1:N]';
            return;
        end
    end
    resampled_indices(i,1)=j;
    i=i+1;
end
end