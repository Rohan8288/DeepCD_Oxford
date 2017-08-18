clear all;

run('./vlfeat/toolbox/vl_setup.m');

data_dir = './data/';
data_number = 8;
data_name = {'bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall'};
image_number = 6;
sample_point = 5000;

for dn = 1:data_number
    data_path = [data_dir, data_name{dn}, '/'];
    patch_path = [data_path, 'patch/'];
    
    if ~exist(patch_path, 'dir')
       mkdir(patch_path); 
    end
    
    for in = 1:image_number
        fprintf('process %d %d\n', dn, in);
        descriptor_path = [patch_path, num2str(in), '/'];
%        if exist(descriptor_path, 'dir')
%           rmdir(descriptor_path); 
%      end
        mkdir(descriptor_path); 
        
        image_path = [data_path, 'img', num2str(in), '.ppm'];
        image = imread(image_path);
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
        
        if size(frame, 2) > sample_point
            [~, sort_index] = sort(abs(info.peakScores), 'descend');
            frame = frame(:, sort_index(1:sample_point));
            patch = patch(:, sort_index(1:sample_point));
        end
        
        [~, descriptor] = vl_covdet(image, 'frames', frame, 'descriptor', 'SIFT', ...
                                    'PatchResolution', 31, 'OctaveResolution', 127, ...
                                    'DoubleImage', true, 'estimateOrientation', true, ...
                                    'estimateAffineShape', true);
        save([descriptor_path, 'R_64_Sift.mat'], 'frame', 'descriptor');
        
        patch = reshape(patch, 63, 63, size(patch, 2));
        patch_number = size(patch, 3);
        local_norm_patch_32 = zeros(32, 32, 1, patch_number);
        local_norm_patch_64 = zeros(64, 64, 1, patch_number);
        for pn = 1:patch_number
            tmp_patch = imresize(patch(:, :, pn), [32, 32]);
            tmp_patch = (tmp_patch - mean2(tmp_patch)) / std2(tmp_patch);
            local_norm_patch_32(:, :, :, pn) = tmp_patch;
            tmp_patch = imresize(patch(:, :, pn), [64, 64]);
            tmp_patch = (tmp_patch - mean2(tmp_patch)) / std2(tmp_patch);
            local_norm_patch_64(:, :, :, pn) = tmp_patch;
        end
        
        save([descriptor_path, 'R_64_patch.mat'], 'frame', 'local_norm_patch_32', 'local_norm_patch_64');
    end
end
