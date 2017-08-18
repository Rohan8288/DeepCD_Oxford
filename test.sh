DATAALL=(bark bikes boat graf leuven trees ubc wall)

DATANAME=${DATAALL[7]}
DATANUMBER=6
NORMALIZETYPE=1

RESOLUTION=(64)
NETWORKTYPE=(TFeat_M)


for ((j=0; j<${#RESOLUTION[@]}; j=j+1))
do
	for ((i=0; i<${#NETWORKTYPE[@]}; i=i+1))
	do
		th extractOther.lua -dataName $DATANAME -dataNumber $DATANUMBER -resolution ${RESOLUTION[$j]} -networkType ${NETWORKTYPE[$i]} -normalizeType $NORMALIZETYPE
	done
done
