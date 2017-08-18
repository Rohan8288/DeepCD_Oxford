require 'xlua'
require 'trepl'
require 'image'
require 'nn'
require 'torch'
require 'mattorch'
require 'stn'
local pl = require('pl.import_into')()

local cmd = torch.CmdLine()
cmd:option('-dataName', 'fountain')
cmd:option('-dataNumber', 11)
cmd:option('-resolution', 64)
cmd:option('-batchSize', 128)
cmd:option('-modelDirectory', '../CPU/') 
cmd:option('-networkType', '') -- necessary
cmd:option('-normalizeType', 1)
local option = cmd:parse(arg)
assert(option.netWork ~= '', ' you should specify the network type')
local data_name = option.dataName
local data_number = option.dataNumber
local resolution = option.resolution
local batch_size = option.batchSize
local model_directory = option.modelDirectory
local network_type = option.networkType
local normalize_type = option.normalizeType
local descriptor_dimension = 128

local model
local input_patch_size
if network_type == 'DeepDesc_a' then
	model = 'DeepDesc_all.t7'
	input_patch_size = 64
elseif network_type == 'DeepDesc_ly' then
	model = 'DeepDesc_liberty+yosemite.t7'
	input_patch_size = 64
elseif network_type == 'PNNet' then
	model = 'PNNet_liberty.t7'
	input_patch_size = 32
elseif network_type == 'TFeat_R' then
	model = 'TFeat_RatioS_liberty.t7'
	input_patch_size = 32
elseif network_type == 'TFeat_M' then
	model = 'TFeat_MarginS_liberty.t7'
	input_patch_size = 32
else
	print(' the model hasnt been defined')
	os.exit()
end

local net = torch.load(model_directory..model)
print(net)

for image = 1,data_number do
	print(' image: '..image)

	-------------------------------------------------------------------------
	--Load patch-------------------------------------------------------------
	local file_content = mattorch.load('./data/'..data_name..'/patch/'..image..'/R_'..resolution..'_patch.mat')
	--Due to the mattorch format, the loaded matrix should be transposed
	--Remember to use "clone()" inside the for loop
	local frame = file_content.frame:clone()
	local patch
	
	if normalize_type == 0 then
		if input_patch_size == 32 then
			patch = file_content.patch_32:clone()
		elseif input_patch_size == 64 then
			patch = file_content.patch_64:clone()
		else 
			print(' you should first extract patch with suitable size!')
			os.exit()
		end
	elseif normalize_type == 1 then
		if input_patch_size == 32 then
			patch = file_content.local_norm_patch_32:clone()
		elseif input_patch_size == 64 then
			patch = file_content.local_norm_patch_64:clone()
		else 
			print(' you should first extract patch with suitable size!')
			os.exit()
		end
	elseif normalize_type == 2 then
		if input_patch_size == 32 then
			patch = file_content.global_norm_patch_32:clone()
		elseif input_patch_size == 64 then
			patch = file_content.global_norm_patch_64:clone()
		else 
			print(' you should first extract patch with suitable size!')
			os.exit()
		end
	else
		print(' invalid normalization type')
		os.exit()
	end

	local patch_number = patch:size(1)

	for k = 1,patch_number do
		local tmp = patch[{k, 1, {}, {}}]:clone()
		patch[{k, 1, {}, {}}] = tmp:t():clone()
	end
	patch = patch:float() -- This is an important step since net:forward can only input FloatTensor

	-------------------------------------------------------------------------
	--Feed into network------------------------------------------------------
	local descriptor = torch.Tensor(patch_number, descriptor_dimension)
	local descriptor_split = descriptor:split(batch_size)

	for i,v in ipairs(patch:split(batch_size)) do
		descriptor_split[i]:copy(net:forward(v))
	end

	-------------------------------------------------------------------------
	--Save the output--------------------------------------------------------
	local output_content = {frame = frame, descriptor = descriptor}
	mattorch.save('./data/'..data_name..'/patch/'..image..'/R_'..resolution..'_'..network_type..'.mat', output_content)

	collectgarbage()
end
