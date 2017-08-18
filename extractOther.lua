require 'cudnn'
require 'cunn'
require 'image'
require 'mattorch'
require 'nn'
require 'stn'
require 'trepl'
require 'xlua'
local pl = require('pl.import_into')()

local cmd = torch.CmdLine()
cmd:option('-dataName', '')
cmd:option('-imageNum', 6)
cmd:option('-batchSize', 128)
cmd:option('-modelDir', './model/') 
cmd:option('-networkType', '') -- necessary
local option = cmd:parse(arg)
assert(option.networkType ~= '', ' you should specify the network type')

local dataName = option.dataName
local imageNum = option.imageNum
local batchSize = option.batchSize
local modelDir = option.modelDir
local networkType = option.networkType
local descriptorDim = 128

local model
local inputPatchSize
if networkType == 'DeepDesc_a' then
	model = 'DeepDesc_all.t7'
	inputPatchSize = 64
elseif networkType == 'DeepDesc_ly' then
	model = 'DeepDesc_liberty+yosemite.t7'
	inputPatchSize = 64
elseif networkType == 'PNNet' then
	model = 'PNNet_liberty.t7'
	inputPatchSize = 32
elseif networkType == 'TFeat_R' then
	model = 'TFeat_RatioS_liberty.t7'
	inputPatchSize = 32
elseif networkType == 'TFeat_M' then
	model = 'TFeat_MarginS_liberty.t7'
	inputPatchSize = 32
else
	print(' the model type hasnt been defined')
	os.exit()
end

local network = torch.load(paths.concat(modelDir, model)):cuda()
print('use '..networkType..' to extract the feature of '..dataName)

for image = 1,imageNum do
	print(' image: '..image)

	-------------------------------------------------------------------------
	--Load patch-------------------------------------------------------------
	local fileContent = mattorch.load(paths.concat('gooddata', dataName, 'patch', image, 'R_64_patch.mat'))
	--Due to the mattorch format, the loaded matrix should be transposed
	--Remember to use "clone()" inside the for loop
	local frame = fileContent.frame:clone()
	local patch
	
	if inputPatchSize == 32 then
		patch = fileContent.local_norm_patch_32:clone()
	elseif inputPatchSize == 64 then
		patch = fileContent.local_norm_patch_64:clone()
	else 
		print(' you should first extract patch with suitable size!')
		os.exit()
	end

	local patchNum = patch:size(1)

	for k = 1,patchNum do
		local tmp = patch[{k, 1, {}, {}}]:clone()
		patch[{k, 1, {}, {}}] = tmp:t():clone()
	end
	patch = patch:float() -- This is an important step since net:forward can only input FloatTensor

	-------------------------------------------------------------------------
	--Feed into network------------------------------------------------------
	local descriptor = torch.Tensor(patchNum, descriptorDim)
	local descriptorSplit = descriptor:split(batchSize)

	for i,v in ipairs(patch:split(batchSize)) do
		v = v:cuda()
		descriptorSplit[i]:copy(network:forward(v))
	end

	-------------------------------------------------------------------------
	--Save the output--------------------------------------------------------
	local outputContent = {frame = frame, descriptor = descriptor}
	mattorch.save(paths.concat('gooddata', dataName, 'patch', image, 'R_64_'..networkType..'.mat'), outputContent)

	collectgarbage()
end
