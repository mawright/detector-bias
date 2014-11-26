function [m_hat, c_hat] = DTbiasEstimation( Netflow, Density )

notConverged = true;

nTime = numel(Netflow);

Gamma = .005;
x = linspace(-3,3,30);
Phik = 1/sqrt(2*pi)*exp(-x.^2)';

Phi = buildKernelMatrix( Phik, nTime);

c_hat = zeros(numel(Netflow),1);
m_hat = zeros(size(c_hat));

stoppingThreshold = .01;
iterCount = 0;

while notConverged
    
    lastm = m_hat;
    
    for t = 1:nTime-1
        
        n_apriori(t+1) = Density(t) + m_hat(t) + Netflow(t);
        
        n_tilde_apriori(t+1) = n_apriori(t+1) - Density(t+1);
        
        n_tilde_aposteriori(t+1) = n_tilde_apriori(t+1) / (1 + Phi(:,t)' * Gamma * Phi(:,t));
        
%         n_tilde_aposteriori(t+1) = update_n_tilde( n_tilde_apriori(t+1), Phi(:,t), Gamma );
        
%         c_hat = update_c_hat(c_hat, Gamma, Phi(:,t), n_tilde_aposteriori(t+1));
        
        c_hat = c_hat + Gamma * Phi(:,t) * n_tilde_aposteriori(t+1);
        m_hat(t) = Phi(:,t)' * c_hat;
                
    end
        
    if norm(m_hat-lastm) < stoppingThreshold
        notConverged = false;
    end
    iterCount = iterCount + 1;
    fprintf('Iter %g: norm of m difference vector is %.5f\n', iterCount, norm(m_hat-lastm));
    
    figure(1);
    plot(1:length(m_hat),m_hat,1:length(lastm),lastm,1:length(Netflow),Netflow);
    legend('Current estimated bias','Previous iteration''s bias estimate','Net flow difference');
    figure(2);
%     plot(1:length(n_tilde_aposteriori),n_tilde_aposteriori,1:length(n_apriori),n_apriori,1:length(Density),Density);
%     legend('n error','nhat','measured n');
    plot(1:length(n_apriori), n_apriori,1:length(Density),Density);
    legend('nhat','Measured n');
    pause(.1);
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

function n_tilde_aposteriori = update_n_tilde( n_tilde_apriori, Phik, Gamma )
n_tilde_aposteriori = n_tilde_apriori / (1 + Phik' * Gamma * Phik);
end

function c_hat = update_c_hat( c_hat_in, Gamma, Phik, n_tilde_aposteriori )
c_hat = c_hat_in + Gamma * Phik * n_tilde_aposteriori;
end