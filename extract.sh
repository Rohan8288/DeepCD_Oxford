DATAALL=(bark bikes boat graf leuven trees ubc wall)

IMAGENUMBER=6

NETWORKTYPE=(TFeat_M TFeat_R PNNet DeepDesc_a DeepDesc_ly)
NETWORKTYPE2=(DeepCD_2S DeepCD_2S_noSTN DeepCD_Sp DeepCD_2S_new)

for ((j=0; j<${#DATAALL[@]}; j=j+1))
do
	for ((i=0; i<${#NETWORKTYPE[@]}; i=i+1))
	do
		th extractOther.lua -dataName ${DATAALL[$j]} -imageNum $IMAGENUMBER -networkType ${NETWORKTYPE[$i]}
	done
	for ((i=0; i<${#NETWORKTYPE2[@]}; i=i+1))
	do
		th extractDeepCD.lua -dataName ${DATAALL[$j]} -imageNum $IMAGENUMBER -networkType ${NETWORKTYPE2[$i]}
	done
done
