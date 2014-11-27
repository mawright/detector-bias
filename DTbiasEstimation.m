function [m_hat, c_hat] = DTbiasEstimation( Netflow, Density )

notConverged = true;

nTime = numel(Netflow);

Gamma = .05;
x = linspace(-3,3,1000);
Phik = 1/sqrt(2*pi)*exp(-x.^2)';

Phi = buildKernelMatrix( Phik, nTime);

c_hat = zeros(numel(Netflow),1);
m_hat = zeros(size(c_hat));

stoppingThreshold = .07;
iterCount = 0;

while notConverged
    
    lastm = m_hat;
    
    for t = 1:nTime-1
        
        nonzeroKernelIndeces = Phi(:,t) ~= 0;
        
        n_apriori(t+1) = Density(t) + m_hat(t) + Netflow(t);
        
        n_tilde_apriori(t+1) = Density(t+1) - n_apriori(t+1);
        
        n_tilde_aposteriori(t+1) = n_tilde_apriori(t+1) / ...
                (1 + Phi(nonzeroKernelIndeces,t)' * Gamma * Phi(nonzeroKernelIndeces,t));
        
        c_hat(nonzeroKernelIndeces) = c_hat(nonzeroKernelIndeces) +...
                Gamma * Phi(nonzeroKernelIndeces,t) * n_tilde_aposteriori(t+1);
        m_hat(t) = Phi(nonzeroKernelIndeces,t)' * c_hat(nonzeroKernelIndeces);
                
    end
        
    if norm(m_hat-lastm) < stoppingThreshold
        notConverged = false;
    end
    iterCount = iterCount + 1;
    fprintf('Iter %g: norm of m difference vector is %.5f\n', iterCount, norm(m_hat-lastm));
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