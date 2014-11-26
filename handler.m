function [ RegressedSignal, BiasSignal ] = handler( LoopData, PositiveFlowDetectors, NegativeFlowDetectors )

Inflow = 0;
Outflow = 0;

if strcmpi(LoopData.Units,'SI')
    timeConversionFactor = 1;
    distanceConversionFactor = 1609.34; % meters in 1 mile
end

for i = 1:length(PositiveFlowDetectors)
    DetectorIndex = LoopData.OriginalVdsIds == PositiveFlowDetectors{i};
    Inflow = Inflow + LoopData.Flows(DetectorIndex,:) * timeConversionFactor;
end

for i = 1:length(NegativeFlowDetectors)
    DetectorIndex = LoopData.OriginalVdsIds == NegativeFlowDetectors{i};
    Outflow = Outflow + LoopData.Flows(DetectorIndex,:) * timeConversionFactor;
end

Inflow = interpolateForNans( Inflow );
Outflow = interpolateForNans( Outflow );

DensityDetectorIndex = LoopData.OriginalVdsIds == PositiveFlowDetectors{1};
Density = LoopData.Density(DensityDetectorIndex,:);
Density = interpolateForNans( Density );

% Hardcoded
UpstreamPostmile = 30.14; % detector 717661
DownstreamPostmile = 30; % detector 717657

Length = abs( DownstreamPostmile - UpstreamPostmile ) * distanceConversionFactor;

Occupancy = Density * Length;

Netflow = Inflow - Outflow;

NetflowFiner = interp1(linspace(0,24*3600,length(Netflow)),Netflow,0:5:24*3600,'spline');
OccupancyFiner = interp1(linspace(0,24*3600,length(Occupancy)),Occupancy,0:5:24*3600,'spline');
[ RegressedSignal, BiasSignal ] = DTbiasEstimation( NetflowFiner, OccupancyFiner );
end

function filledInTimeSeries = interpolateForNans( inputTimeSeries )
nanindeces = isnan(inputTimeSeries);
interpedvalues = interp1( find(~nanindeces), inputTimeSeries(~nanindeces), find(nanindeces));

filledInTimeSeries = inputTimeSeries;
filledInTimeSeries(nanindeces) = interpedvalues;
end