function [ RegressedSignal, BiasSignal ] = handler( pemsfilename, PositiveFlowDetectors, NegativeFlowDetectors )

pems = load(pemsfilename);

if strcmpi(pems.pems.datatype,'pems5min')
    timeConversionFactor = 5/60;
end

for i = 1:length(PositiveFlowDetectors)
    DetectorIndex = pems.pems.vds == PositiveFlowDetectors{i};
    Inflow = Inflow + sum(pems.pems.data(DetectorIndex).flw,2) / timeConversionFactor;
end

for i = 1:length(NegativeFlowDetectors)
    DetectorIndex = pems.pems.vds == NegativeFlowDetectors{i};
    Outflow = Outflow + sum(pems.pems.data(DetectorIndex).flw,2) / timeConversionFactor;
end

DensityDetectorIndex = pems.pems.vds == PositiveFlowDetectors{1};
Density = sum(pems.pems.data(DensityDetectorIndex).dty,2);

% Hardcoded
UpstreamPostmile = 35.41;
DownstreamPostmile = 34.9;

Length = abs( DownstreamPostmile - UpstreamPostmile );

Occupancy = Density * Length;

[ RegressedSignal, BiasSignal ] = DTbiasEstimation( Inflow - Outflow, Occupancy );