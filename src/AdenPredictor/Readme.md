# AdenPredictor User Guide v.1.0.1
Written by Romel Baral, Mihir Mongia, Hosein Mohimani

## About AdenPredictor
AdenPredictor is a tool to predict substrate specificity of Adenylation domains of Non-Ribosomal PeptideSynthetases (NRPSs) and the biosynthetic gene clusters encoding them.

## Input and output
AdenPredictor takes raw fasta files or tsv files as input, and outputs a tsv file with all the predictions. It can predict at most 21 substrates per input fasta sequence.

If fasta format is used, input is processed by hmmpfam2 first, then AdenPredictor predicts the substrates based on a variety of python based machine learning classifiers. One hot encoding style is used for data encoding. If tsv format is used, hmmpfam2 step is skipped.

## Input Parameters
AdenPredictor has one necessary argument: -i. Rest of the arguments are optional. The options are listed below.  
**-h** is the option for help. It prints brief help manual.  
**-i** option indicate the absolute or relative input file path. It can be either fasta format or tsv. Default input option is fasta format.  
**-o** option indicates output filepath. As with input option -i, output path can be relative or absolute. If no output path is specified, a file is created with the similar name as input. This output file is named based on timestamp of run.  
**-s** option indicates input type. There are two valid options, 0 (default), and 1. Option 0 indicates that the input consisting of amino acid sequences is in fasta format. Option 1 indicates that the input consisting of extracted signatures is in tsv format. Tsv file must be headerless annd consisting of two columns. First column should be 34 amino acid long signature and second column should be id of the signature.  
**-k** is the number of predictions for substrate amino acid. Default value of k is 1. Because of the training dataset, as of now, k can be at most 21.

## Output Format
Output is a tsv file. The first two columns are sequence Id and Signature. Rest k columns denote the predicted sustrates arranged in decreasing order of importance. The Ids are extracted from the input file.

## Softwares/Packages required
1. Python: 3.5 and onwards. Required packages are **[numpy](https://numpy.org/install/)**, **[sklearn](https://scikit-learn.org/stable/install.html)**. Follow the hyperlinks to install individual modules.
2. hmmpfam2

## How to Run:
Here are a few sample commands. Remember, -i is the only manadatory option.

./AdenPredictor -h  
./AdenPredictor -i \<input file>  
./AdenPredictor -i \<input file> -o \<output file> -s 0  
./AdenPredictor -i \<input file> -o \<output file> -s 1  
./AdenPredictor -i \<input file> -o \<output file> -k \<number of substrate predictions>  
./AdenPredictor -i \<input file> -o \<output file> -k \<number of substrate predictions> -s 0  
./AdenPredictor -i \<input file> -o \<output file> -k \<number of substrate predictions> -s 1

