clear all

%% Library and paths
run ./vlfeat/toolbox/vl_setup
data_dir = './data/';
data_name = {'bark', 'bikes', 'boat', 'graf', 'leuven', 'trees', 'ubc', 'wall'};


%% Important parameters

isInter = 0; % Default 0. If you set to 1 then the interpolation will be activated.
des = 'sift'; % There are three type descriptor in vlfeat covdet. You can choose 'sift', 'liop', or 'patch'.
opt.sc_min = 1/6; % The smallest size.
opt.sc_max = 1.5; % The biggest size.
opt.ns = 10; % Number of the sampled scales.

%% (Optional, usually you don't want to change these.)
% The following parameters belongs to the extended version of ASV.
% While scale space is studied in the original setting,
% rotation might also help to improve the performace.
% We do not change any of the rotation parameters for convience,
% but you are free to try these. The performance will be further improved.
opt.rc_min = 0; % The smallest angle.
opt.rc_max = 0; % The biggest angle.
opt.nr = 1; % Number of the sampled angles.

data_number = 8;
image_number = 6;


%% Extract the descriptor from the whole dataset
for dn = 1:data_number
    for in = 1:image_number
        fprintf('dn:%d  in:%d\n',dn,in)
        image = imread([data_dir, data_name{dn}, '/img', num2str(in), '.ppm']);

        if size(image,3)>1
            image = rgb2gray(image);
        end
        image = single(image);
        
        content = load([data_dir, data_name{dn}, '/patch/', num2str(in), '/R_64_Sift.mat']);
        frame = content.frame;
        des = content.descriptor;
        
        %% ASV(1S): median thresholding
        descriptor = vl_asvcovdet(image, opt, frame, 'sift', isInter);
        save([data_dir, data_name{dn}, '/patch/', num2str(in), '/R_64_ASVSift.mat'],'frame','descriptor')
    end
end





