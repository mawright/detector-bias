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
        
%         n_tilde_aposteriori(t+1) = update_n_tilde( n_tilde_apriori(t+1), Phi(:,t), Gamma );
        
%         c_hat = update_c_hat(c_hat, Gamma, Phi(:,t), n_tilde_aposteriori(t+1));
        
        c_hat(nonzeroKernelIndeces) = c_hat(nonzeroKernelIndeces) +...
                Gamma * Phi(nonzeroKernelIndeces,t) * n_tilde_aposteriori(t+1);
        m_hat(t) = Phi(nonzeroKernelIndeces,t)' * c_hat(nonzeroKernelIndeces);
                
    end
    
    if iterCount == 10
        mhat_at_10 = m_hat;
        ntilde_at_10 = n_tilde_aposteriori;
    elseif iterCount == 30
        mhat_at_30 = m_hat;
        ntilde_at_30 = n_tilde_aposteriori;
    elseif iterCount == 50
        mhat_at_50 = m_hat;
        ntilde_at_50 = n_tilde_aposteriori;
    elseif iterCount == 70
        mhat_at_70 = m_hat;
        ntilde_at_70 = n_tilde_aposteriori;
    end
        
    if norm(m_hat-lastm) < stoppingThreshold
        notConverged = false;
    end
    iterCount = iterCount + 1;
    fprintf('Iter %g: norm of m difference vector is %.5f\n', iterCount, norm(m_hat-lastm));
    
%     figure(1);
%     plot(1:length(m_hat),m_hat,1:length(lastm),lastm,1:length(Netflow),Netflow);
%     legend('Current estimated bias','Previous iteration''s bias estimate','Net flow difference');
%     figure(2);
% %     plot(1:length(n_tilde_aposteriori),n_tilde_aposteriori,1:length(n_apriori),n_apriori,1:length(Density),Density);
% %     legend('n error','nhat','measured n');
% %     plot(1:length(n_apriori), n_apriori,1:length(Density),Density+Netflow);
% %     legend('nhat','n');
%     plot(n_tilde_aposteriori); legend('ntilde');
%     pause(.1);
end

figure(1);
plot(1:length(Netflow),-Netflow*300,1:length(mhat_at_10),mhat_at_10*300,...
     1:length(mhat_at_30),mhat_at_30*300,1:length(mhat_at_50),mhat_at_50*300,...
     1:length(mhat_at_70),mhat_at_70*300);
legend('Net detector flow difference','Estimated bias (iteration 10)','Iteration 30','Iteration 50','Iteration 70');
ylabel('Vehicles/hr');

set(gca,'XTick',0:3*3600/5:24*3600/5);
set(gca,'XTickLabel',{'0','3','6','9','12','15','18','21','24'});

figure(2);
% plot(1:length(Density),Density,1:length(nhat_at_10),nhat_at_10,...
%      1:length(nhat_at_30),nhat_at_30,1:length(nhat_at_50),nhat_at_50,...
%      1:length(nhat_at_70),nhat_at_70);
% legend('Measured detector density','Estimated density (iteration 10)','Iteration 30','Iteration 50','Iteration 70');
% ylabel('Vehicles');

plot(1:length(ntilde_at_10),ntilde_at_10,...
     1:length(ntilde_at_30),ntilde_at_30,1:length(ntilde_at_50),ntilde_at_50,...
     1:length(ntilde_at_70),ntilde_at_70);
legend('Density error (iteration 10)','Iteration 30','Iteration 50','Iteration 70');
ylabel('Vehicles');

set(gca,'XTick',0:3*3600/5:24*3600/5);
set(gca,'XTickLabel',{'0','3','6','9','12','15','18','21','24'});

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