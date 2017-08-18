DATAALL=(bark bikes boat graf leuven trees ubc wall)

DATANAME=${DATAALL[7]}
DATANUMBER=6
NORMALIZETYPE=1

RESOLUTION=(64)
NETWORKTYPE=(TFeat_M TFeat_R PNNet DeepDesc_a DeepDesc_ly)
NETWORKTYPE2=(DeepCD_2S DeepCD_2S_noSTN DeepCD_Sp)

for ((j=0; j<${#RESOLUTION[@]}; j=j+1))
do
	for ((i=0; i<${#NETWORKTYPE[@]}; i=i+1))
	do
		th extractOther.lua -dataName $DATANAME -dataNumber $DATANUMBER -resolution ${RESOLUTION[$j]} -networkType ${NETWORKTYPE[$i]} -normalizeType $NORMALIZETYPE
	done
	for ((i=0; i<${#NETWORKTYPE2[@]}; i=i+1))
	do
		th extractDeepCD.lua -dataName $DATANAME -dataNumber $DATANUMBER -resolution ${RESOLUTION[$j]} -networkType ${NETWORKTYPE2[$i]} -normalizeType $NORMALIZETYPE
	done
done
