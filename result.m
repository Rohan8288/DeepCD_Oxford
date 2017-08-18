run('./vlfeat/toolbox/vl_setup.m');
import benchmarks.*;

option.dataDir = './gooddata/';
option.dataName = {'bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall'};
%option.isPPLF = 0; % Prior Product Late Fusion
option.isLRC = 0;
option.isRT = 1;
option.dataNumber = 8;
option.imageNumber = 6;
%option.pplfGamma = 1;

networkType = {'Sift', 'DeepDesc_ly', 'DeepDesc_a', 'PNNet', 'TFeat_R', 'TFeat_M', 'DeepCD_2S', 'DeepCD_2S_noSTN', 'DeepCD_2S_new'};
networkNum = size(networkType, 2);
apArray = zeros(1, option.dataNumber);
correctMatch = cell(networkNum, 1);

for i = 1:networkNum
    option.networkType = networkType{i};
    [ap, correct, precision, recall] = evaluation(option);
	
	apArray(i, :) = mean(ap, 2);
    correctMatch{i} = correct;
end

map = mean(apArray');

for i = 1:networkNum
	fprintf('%s: %f\n', networkType{i}, map(i));
end
