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
assert(option.networkType ~= '', 'you should specify the network type')

local dataName = option.dataName
local imageNum = option.imageNum
local batchSize = option.batchSize
local modelDir = option.modelDir
local networkType = option.networkType

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

local network = torch.load(paths.concat(modelDir, model)):cuda()

-----------------------------------------------------------------------------
--Remove mulconstant---------------------------------------------------------
if networkType == 'DeepCD_2S' then
	network:get(1):get(2):remove(13)
elseif networkType == 'DeepCD_2S_new' then
	network:get(1):get(2):remove(12)
elseif networkType == 'DeepCD_2S_noSTN' then
	network:get(1):get(2):remove(12)
elseif networkType == 'DeepCD_Sp' then
	network:get(7):get(2):remove(6)
end
print('use '..networkType..' to extract the feature of '..dataName)

for image = 1,imageNum do
	print(' image: '..image)

	for n = 1,2 do
		---------------------------------------------------------------------
		--Select table-------------------------------------------------------
		if n == 1 then
			if image ~= 1 then
				if networkType == 'DeepCD_2S' then
					network:remove(2)
				elseif networkType == 'DeepCD_2S_new' then
					network:remove(2)
				elseif networkType == 'DeepCD_2S_noSTN' then
					network:remove(2)
				elseif networkType == 'DeepCD_Sp' then
					network:remove(8)
				end
			end

			network:add(nn.SelectTable(1))
		else
			if networkType == 'DeepCD_2S' then
				network:remove(2)
			elseif networkType == 'DeepCD_2S_new' then
				network:remove(2)
			elseif networkType == 'DeepCD_2S_noSTN' then
				network:remove(2)
			elseif networkType == 'DeepCD_Sp' then
				network:remove(8)
			end
			
			network:add(nn.SelectTable(2))
		end

		---------------------------------------------------------------------
		--Load patch---------------------------------------------------------
		local fileContent = mattorch.load(paths.concat('gooddata', dataName, 'patch', image, 'R_64_patch.mat'))
		--Due to the mattorch format, the loaded matrix should be transposed
		--Remember to use "clone()" inside the for loopi
		frame = fileContent.frame:clone()
		local patch = fileContent.local_norm_patch_32:clone()

		local patchNum = patch:size(1)	

		for t = 1,patchNum do
			local tmp = patch[{t, 1, {}, {}}]:clone()
			patch[{t, 1, {}, {}}] = tmp:t():clone() 
		end

		patch = patch:float() -- This is an important step since net:forward can only input FloatTensor
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
			v = v:cuda()
			if n == 1 then
				descriptorSplit[i]:copy(network:forward(v))
			else
				descriptorSplit[i]:copy(network:forward(v))
			end
		end
		
		-------------------------------------------------------------------------
		--Save the output--------------------------------------------------------
		if n == 1 then
			descriptorLead = descriptor:clone()
		else
			local descriptorComplete = descriptor:clone()
			local outputContent = {frame = frame, descriptor_lead = descriptorLead, descriptor_complete = descriptorComplete}
			mattorch.save(paths.concat('gooddata', dataName, 'patch', image, 'R_64_'..networkType..'.mat'), outputContent)
		end

		collectgarbage()
	end
end
