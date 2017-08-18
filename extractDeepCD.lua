local xlua = require 'xlua'
--require 'trepl'
--require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'

local mattorch = require 'mattorch'

local pl = require('pl.import_into')()

local cmd = torch.CmdLine()
cmd:option('-dataName', 'fountain')
cmd:option('-dataNumber', 11)
cmd:option('-resolution', 64)
cmd:option('-batchSize', 128)
cmd:option('-modelDirectory', '../GPU/')
cmd:option('-networkType', '') -- necessary
cmd:option('-normalizeType', 1)
cmd:option('-gpu', 'true')
local option = cmd:parse(arg)
assert(option.netWork ~= '', 'you should specify the network type')
local dataName = option.dataName
local dataNum = option.dataNumber
local resolution = option.resolution
local batchSize = option.batchSize
local modelDir = option.modelDirectory
local networkType = option.networkType
local normalizeType = option.normalizeType

local model
if networkType == 'DeepCD_2S' then
	model = 'DeepCD_2Stream_liberty.t7'
elseif networkType == 'DeepCD_2S_new' then
	model = 'DeepCD_2S_DDU_liberty.t7'
elseif networkType == 'DeepCD_2S_noSTN' then
	model = 'DeepCD_noSTN_2Stream.t7'
elseif networkType == 'DeepCD_Sp' then
	model = 'DeepCD_Split_liberty.t7'
else 
	print(' you should define model path first')
	os.exit()
end

local net = torch.load(paths.concat(modelDir, model))
if option.gpu then
	net = net:cuda()
end
-----------------------------------------------------------------------------
--Remove mulconstant---------------------------------------------------------
if networkType == 'DeepCD_2S' then
	net:get(1):get(2):remove(13)
elseif networkType == 'DeepCD_2S_new' then
	net:get(1):get(2):remove(12)
elseif networkType == 'DeepCD_2S_noSTN' then
	net:get(1):get(2):remove(12)
elseif networkType == 'DeepCD_Sp' then
	net:get(7):get(2):remove(6)
end
print(net)

for image = 1,dataNum do
	print(' image: '..image)

	for n = 1,2 do
		---------------------------------------------------------------------
		--Select table-------------------------------------------------------
		if n == 1 then
			if image ~= 1 then
				if networkType == 'DeepCD_2S' then
					net:remove(2)
				elseif networkType == 'DeepCD_2S_new' then
					net:remove(2)
				elseif networkType == 'DeepCD_2S_noSTN' then
					net:remove(2)
				elseif networkType == 'DeepCD_Sp' then
					net:remove(8)
				end
			end

			net:add(nn.SelectTable(1))
		else
			if networkType == 'DeepCD_2S' then
				net:remove(2)
			elseif networkType == 'DeepCD_2S_new' then
				net:remove(2)
			elseif networkType == 'DeepCD_2S_noSTN' then
				net:remove(2)
			elseif networkType == 'DeepCD_Sp' then
				net:remove(8)
			end
			
			net:add(nn.SelectTable(2))
		end

		---------------------------------------------------------------------
		--Load patch---------------------------------------------------------
		local fileContent = mattorch.load('./gooddata/'..dataName..'/patch/'..image..'/R_'..resolution..'_patch.mat')
		--Due to the mattorch format, the loaded matrix should be transposed
		--Remember to use "clone()" inside the for loopi
		frame = fileContent.frame:clone()
		local patch
		
		if normalizeType == 0 then
			patch = fileContent.patch_32:clone()
		elseif normalizeType == 1 then
			patch = fileContent.local_norm_patch_32:clone()
		elseif normalizeType == 2 then
			patch = fileContent.global_norm_patch_32:clone()
		else
			print(' invalid normalization type')
			os.exit()
		end

		local patchNum = patch:size(1)	

		for t = 1,patchNum do
			local tmp = patch[{t, 1, {}, {}}]:clone()
			patch[{t, 1, {}, {}}] = tmp:t():clone() 
		end

		patch = patch:float() -- This is an important step since net:forward can only input FloatTensor
		if option.gpu then
			patch = patch:cuda()
		end
		---------------------------------------------------------------------
		--Feed into network--------------------------------------------------
		local descriptor
		local descriptorSplit

		if n == 1 then
			descriptor = torch.Tensor(patchNum, 128)
			descriptorSplit = descriptor:split(batchSize)
		else
			descriptor = torch.Tensor(patchNum, 256)
			descriptorSplit = descriptor:split(batchSize)
		end

		for i, v in ipairs(patch:split(batchSize)) do
			if n == 1 then
				descriptorSplit[i]:copy(net:forward(v))
			else
				descriptorSplit[i]:copy(net:forward(v))
			end
		end

		-------------------------------------------------------------------------
		--Save the output--------------------------------------------------------
		if n == 1 then
			descriptorLead = descriptor:clone()
		else
			local descriptorComplete = descriptor:clone()
			local outputContent = {frame = frame, descriptor_lead = descriptorLead, descriptor_complete = descriptorComplete}
			mattorch.save('./gooddata/'..dataName..'/patch/'..image..'/R_'..resolution..'_'..networkType..'.mat', outputContent)
		end

		collectgarbage()
	end
end
