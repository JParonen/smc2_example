function [resampled_indices]=multinomial(w,N)

cdf_weights=cumsum(w);
resampled_indices=zeros(N,1);
i=1;
while i<=N
    r=rand();
    j=1;
    while cdf_weights(j)<r
        j=j+1;
        if j>length(cdf_weights)
            warning('Resampling failed!');
            resampled_indices(:,1)=[1:N]';
            return
        end
    end
    resampled_indices(i,1)=j;
    i=i+1;
end
end