function resampled_indices= systematic(w,N)

%this function implements systematic resampling
% INPUTS:  weigths:  normalized weights
%          numParticles: Number of particles

% OUTPUT: new set of indices

cdf_weights=cumsum(w);
r=rand();
step=1/N;
% divide [0,1] into:
u=[step*r:step:1-(1-r)*step];
resampled_indices=zeros(N,1);
ancestor=1;
i=1;
for j=1:N
   while u(j)>cdf_weights(i)
      if i<N
          i=i+1;
      end
      ancestor=ancestor+1;
      if ancestor>N || i>length(cdf_weights)
          resampled_indices=[1:N]';
          warning('Resampling failed!');
          return;
      end
   end
  resampled_indices(j,1)=ancestor;
end
