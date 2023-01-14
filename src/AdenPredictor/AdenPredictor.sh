#!/bin/bash

printHelp()
{
	echo ""
	echo "Usage:"
	echo "./AdenPredictor -h"
	echo "./AdenPredictor -i <input file> -o <output file> -k <number of substrate predictions>"
	echo "./AdenPredictor -i <input file> -o <output file> -k <number of substrate predictions> -s 0"
	echo "./AdenPredictor -i <input file> -o <output file> -k <number of substrate predictions> -s 1"
	echo "Necessary arguments:"
	echo "-i input file, can be either fasta (default) or tsv"
	echo "Optional argument/flags:"
	echo "-h prints this manual"
	echo "-s if set to 1, assumes that the input is a headerless tsv file"
	echo "The tsv file has two columns, signature and id (in order)"
	echo "If set to default option 0, script assumes that the input is in fasta format"
	echo "-o output filename (relative or absolute path)"
	echo "-k number of substrate amino acid to predict (default value is 1)"
	echo ""
	echo ""
}

illegalOptions=false

while getopts "hi:o:k:s:" opt; do
	case $opt in
		k ) kVal="$OPTARG" ;;
		s ) sVal="$OPTARG" ;;
		i ) tmpFile="$OPTARG" ;;
		o ) outFile="$OPTARG" ;;
		h ) printHelp
		exit ;;
		\? ) illegalOptions=true ;;
	esac
done

if [ -z "$tmpFile" ]; then
	echo "Provide -i option. It is an essential argument"
	illegalOptions=true
fi

if [ "$illegalOptions" = true ]; then
	printHelp
	exit
fi

if [ -f "$tmpFile" ]; then
	tmpFile=$(realpath $tmpFile)
	echo "Absolute path "$tmpFile
else
	echo "File $tmpFile does not exist"
	echo "Please provide correct filepath"
	printHelp
	exit
fi

if [ -z "$sVal" ]; then
	sVal=0
elif [[ $sVal -ne 0 ]] && [[ $sVal -ne 1 ]]; then
	echo "Only 0 or 1 allowed in -s option"
	printHelp
	exit
fi

if [ -z "$kVal" ]; then
	echo "No -k option provided, resorting to default value 1"
	kVal=1
fi

timeString=$(date +_%Y%m%d_%H%M%S_%3N)
pythonScript="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )""/helper/helper.py"

echo "We will extract hmmpfam2 output from "$tmpFile
hmmDatabase="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )""/data/aa-activating.aroundLys.hmm"
hmmFile=$tmpFile$timeString
$(hmmpfam2 $hmmDatabase $tmpFile > $hmmFile)
if [ ! -s $hmmFile ]; then
	echo
	echo
	echo "Invalid format: $tmpFile"
	echo "Please use FASTA format for $tmpFile"
	printHelp
	exit
fi
echo "We have extracted hmmpfam2 output"

# echo "Running python $pythonScript"
if [ -z "$outFile" ] && [[ $sVal -eq 0 ]]; then
	python $pythonScript -i $hmmFile -k $kVal -s $sVal
elif [ -z "$outFile" ]; then
	python $pythonScript -i $tmpFile -k $kVal -s $sVal
elif [[ $sVal -eq 0 ]]; then
	python $pythonScript -i $hmmFile -k $kVal -s $sVal -o $outFile
else
	python $pythonScript -i $tmpFile -k $kVal -s $sVal -o $outFile
fi

$(rm $hmmFile)

