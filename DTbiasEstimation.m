function [biasTimeseries, regressors] = DTbiasEstimation( Netflow, Density )

notConverged = true;

nTime = numel(Netflow);

Gain = .05;
x = linspace(-3,3,1000);
Phik = 1/sqrt(2*pi)*exp(-x.^2)';

Phi = buildKernelMatrix( Phik, nTime);

regressors = zeros(numel(Netflow),1);
biasTimeseries = zeros(size(regressors));

stoppingThreshold = .07;
iterCount = 0;

while notConverged
    
    lastBiasEstimate = biasTimeseries;
    
    for t = 1:nTime-1
        
        nonzeroKernelIndeces = Phi(:,t) ~= 0;
        
        n_apriori(t+1) = Density(t) + biasTimeseries(t) + Netflow(t);
        
        n_tilde_apriori(t+1) = Density(t+1) - n_apriori(t+1);
        
        n_tilde_aposteriori(t+1) = n_tilde_apriori(t+1) / ...
                (1 + Phi(nonzeroKernelIndeces,t)' * Gain * Phi(nonzeroKernelIndeces,t));
        
        regressors(nonzeroKernelIndeces) = regressors(nonzeroKernelIndeces) +...
                Gain * Phi(nonzeroKernelIndeces,t) * n_tilde_aposteriori(t+1);
        biasTimeseries(t) = Phi(nonzeroKernelIndeces,t)' * regressors(nonzeroKernelIndeces);
                
    end
        
    if norm(biasTimeseries-last) < stoppingThreshold
        notConverged = false;
    end
    iterCount = iterCount + 1;
    fprintf('Iter %g: norm of m difference vector is %.5f\n', iterCount, norm(biasTimeseries-lastBiasEstimate));
end

end

function Phi = buildKernelMatrix(kernel,nTime)

kernelSize = numel(kernel);
Phi = zeros(nTime+2*kernelSize,nTime);
numRows = size(Phi,1);
for i = 1:nTime
    numLeadingZeros = i-1;
    numTrailingZeros = numRows + 1 - i - kernelSize;
    Phi(:,i) = vertcat(zeros(numLeadingZeros,1), kernel, zeros(numTrailingZeros,1));
end
Phi(1:floor(kernelSize/2)+1,:) = [];
Phi(nTime+1:size(Phi,1),:) = [];

normalizingFactorMatrix = repmat(sum(Phi,1),size(Phi,1),1);
Phi = Phi ./ normalizingFactorMatrix;
end