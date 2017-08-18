clear all;

run('./vlfeat/toolbox/vl_setup.m');

dataDir = './gooddata/';
dataNum = 8;
dataName = {'bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall'};
imageNum = 6;
samplePoint = 5000;

for dn = 1:dataNum
    dataPath = [dataDir, dataName{dn}, '/'];
    patchPath = [dataPath, 'patch/'];
    
    if ~exist(patchPath, 'dir')
       mkdir(patchPath); 
    end
    
    for in = 1:imageNum
        fprintf('process %d %d\n', dn, in);
        descriptorPath = [patchPath, num2str(in), '/'];
        mkdir(descriptorPath); 
        
        imagePath = [dataPath, 'img', num2str(in), '.ppm'];
        image = imread(imagePath);
        if size(image, 3) > 1
            image = single(rgb2gray(image));
        else
            image = single(image); 
        end
        
        [frame, patch, info] = vl_covdet(image, 'Method', 'DoG', 'descriptor', 'PATCH', ...
                                         'PatchResolution', 31, 'OctaveResolution', 127, ...
                                         'DoubleImage', true, 'estimateOrientation', true, ...
                                         'estimateAffineShape', true);
        %scale = abs(frame(3, :));      
        %frame(:, scale < geomean(scale)) = [];
        %patch(:, scale < geomean(scale)) = [];
        %info.peakScores(:, scale < geomean(scale)) = [];
        
        if size(frame, 2) > samplePoint
            [~, sortIndex] = sort(abs(info.peakScores), 'descend');
            frame = frame(:, sortIndex(1:samplePoint));
            patch = patch(:, sortIndex(1:samplePoint));
        end
        
        [~, descriptor] = vl_covdet(image, 'frames', frame, 'descriptor', 'SIFT', ...
                                    'PatchResolution', 31, 'OctaveResolution', 127, ...
                                    'DoubleImage', true, 'estimateOrientation', true, ...
                                    'estimateAffineShape', true);
        save([descriptorPath, 'R_64_Sift.mat'], 'frame', 'descriptor');
        
        patch = reshape(patch, 63, 63, size(patch, 2));
        patchNum = size(patch, 3);
        local_norm_patch_32 = zeros(32, 32, 1, patchNum);
        local_norm_patch_64 = zeros(64, 64, 1, patchNum);

        for pn = 1:patchNum
            tmpPatch = imresize(patch(:, :, pn), [32, 32]);
            tmpPatch = (tmpPatch - mean2(tmpPatch)) / std2(tmpPatch);
            local_norm_patch_32(:, :, :, pn) = tmpPatch;
            tmpPatch = imresize(patch(:, :, pn), [64, 64]);
            tmpPatch = (tmpPatch - mean2(tmpPatch)) / std2(tmpPatch);
            local_norm_patch_64(:, :, :, pn) = tmpPatch;
        end
        
        save([descriptorPath, 'R_64_patch.mat'], 'frame', 'local_norm_patch_32', 'local_norm_patch_64');
    end
end
