function new_indeces=Resample(weights,Nparticles,method)

if ~exist('method','var')
    method=0;
end
switch method    
    case 0
        new_indeces=systematic(weights,Nparticles);
    case 1
        new_indeces=multinomial(weights,Nparticles);
    case 2
        new_indeces=residual(weights,Nparticles);
end
end