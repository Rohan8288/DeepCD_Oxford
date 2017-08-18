function [apResult, correctMatch, precisionCell, recallCell] = evaluation(option)
    run('./vlfeat/toolbox/vl_setup.m');
    import benchmarks.*;

    networkType = option.networkType;
%    isPPLF = option.isPPLF; % Prior Product Late Fusion
	isLRC = option.isLRC; % Left Right Consistency
	isRT = option.isRT; % Ratio Test
%    pplfGamma = option.pplfGamma;
    dataDir = option.dataDir;
    dataName = option.dataName;
    dataNum = option.dataNumber;
    imageNum = option.imageNumber;
    
    repBenchmark = RepeatabilityBenchmark('Mode','Repeatability');
    repBenchmark.Opts.overlapError = 0.5;
    
    apResult = zeros(dataNum, imageNum - 1);
	correctMatch = zeros(dataNum, imageNum - 1);	

	precisionCell = cell(dataNum, imageNum);
	recallCell = cell(dataNum, imageNum);

    for dn = 1:dataNum
        dataPath = [dataDir, dataName{dn}, '/'];
        patchPath = [dataPath, 'patch/'];

        image = imread([dataPath, 'img1.ppm']);
        imageSize = size(image);

        content = load([patchPath, '1/R_64_', networkType, '.mat']);
        targetFrame = content.frame;
        if strcmp(networkType, 'DeepCD_2S') || strcmp(networkType, 'DeepCD_Sp') || strcmp(networkType, 'DeepCD_2S_noSTN') || strcmp(networkType, 'DeepCD_2S_new')
            targetDesLead = content.descriptor_lead;
            targetDesComplete = content.descriptor_complete;
        else
            targetDes = content.descriptor;
        end

        for in = 2:imageNum
            fprintf('process %d %d\n', dn, in);
            content = load([patchPath, num2str(in), '/R_64_', networkType, '.mat']);
            sourceFrame = content.frame;
            if strcmp(networkType, 'DeepCD_2S') || strcmp(networkType, 'DeepCD_Sp') || strcmp(networkType, 'DeepCD_2S_noSTN') || strcmp(networkType, 'DeepCD_2S_new')
                sourceDesLead = content.descriptor_lead;
                sourceDesComplete = content.descriptor_complete;
            else
                sourceDes = content.descriptor;
            end

            H1to2 = dlmread([dataPath, 'H1to', num2str(in),'p']);

            [~, ~, bestMatches, ~, targetVisible, sourceVisible] = ...
                repBenchmark.testFeatures(H1to2, imageSize, imageSize, targetFrame, sourceFrame);

            targetVisiblePoint = find(targetVisible > 0);
            sourceVisiblePoint = find(sourceVisible > 0);
            point = find(bestMatches(1, :) ~= 0);
            matchPoint = bestMatches(1, (bestMatches(1, :) ~= 0));
            if strcmp(networkType, 'DeepCD_2S') || strcmp(networkType, 'DeepCD_Sp') || strcmp(networkType, 'DeepCD_2S_noSTN') || strcmp(networkType, 'DeepCD_2S_new')
                distanceMatLead = L2D(targetDesLead(:, targetVisiblePoint), ...
                                      sourceDesLead(:, sourceVisiblePoint));
                distanceMatComplete = L2D(targetDesComplete(:, targetVisiblePoint), ...
                                          sourceDesComplete(:, sourceVisiblePoint));
%                if isPPLF
%                    distanceMat = distanceMatLead .* (distanceMatComplete.^(pplfGamma + (1 - pplfGamma) * distanceMatLead ./ 22.6));
					%distanceMat = distanceMatLead;
%                else
                    distanceMat = distanceMatLead .* distanceMatComplete;
%                end
            else
                distanceMat = L2D(targetDes(:, targetVisiblePoint), ...
                                  sourceDes(:, sourceVisiblePoint));
            end

            distanceMat = distanceMat(point, :);
            distanceMat = distanceMat(:, matchPoint);
			[~, matchIndBackward] = min(distanceMat);
            [sortScore, sortInd] = sort(distanceMat, 2);
			matchIndForward = sortInd(:, 1);
            answer = (1:size(sortScore, 1))';

            if ~isempty(answer)
				if isRT
                	matchScoreForward = sortScore(:, 1) ./ sortScore(:, 2);
				else
					matchScoreForward = sortScore(:, 1);
				end
				
                correctMatchForward = (matchIndForward == answer);
				
				pointNum = size(correctMatchForward, 1);
				if isLRC
					score = [];
					correct = [];
			
					for i = 1:pointNum
						if matchIndBackward(matchIndForward(i)) == i
							score = [score, matchScoreForward(i)];
							correct = [correct, correctMatchForward(i)];
						end
					end		
				else
					score = matchScoreForward;
					correct = correctMatchForward;
				end

				[~, sortInd] = sort(score);
				sortCorrect = correct(sortInd);
                effectivePointNum = length(correct);
                precision = zeros(effectivePointNum, 1);
                recall = zeros(effectivePointNum, 1);

				for i = 1:effectivePointNum
                   precision(i) = sum(sortCorrect(1:i)) / i;
                   recall(i) = sum(sortCorrect(1:i)) / effectivePointNum;
                end
				ap = 0;
				for i = 1:effectivePointNum - 1
					ap = ap + (precision(i) + precision(i + 1)) * (recall(i + 1) - recall(i)) / 2;
                end
                correctMatch(dn, in - 1) = sum(correct);
				precisionCell{dn, in} = precision;
				recallCell{dn, in} = recall;
			else
				ap = 0;
            end
			apResult(dn, in - 1) = ap;
        end
    end
end
