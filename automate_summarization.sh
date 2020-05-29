#!/bin/bash

while getopts ":i:o:p:e:d:" opt; do
  case $opt in
    i) in_path="$OPTARG"
    ;;
    o) out_path="$OPTARG"
    ;;
    p) base_dir="$OPTARG"
    ;;
    e) enc_step="$OPTARG"
    ;;
    d) dec_step="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

in_path=$(echo $in_path | sed 's/\/$//g')
out_path=$(echo $out_path | sed 's/\/$//g')
base_dir=$(echo $base_dir | sed 's/\/$//g')
model_name='summary-model'
vocab_name='summary-vocab'
export CLASSPATH=$base_dir/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar


if [[ -d $in_path ]]
then
    echo "Input path is a dir"
    in_type="dir"
else
    echo "Input path is a file"
    in_type="file"
fi

if [[ -d $out_path ]]
then
    echo "Output path is a dir"
    out_type="dir"
else
    echo "Output path is a file"
    out_type="file"
fi

if [[ $in_type != $out_type ]]
then
echo "Input and output types should be similar (file/directory). Exiting with error"
exit 1
fi

# Remove the temp folder (sentence_per_line, transcripts_tokenized, finished_files)
rm -r $base_dir/experiments/temp

if [[ $in_type == "file" ]]
then
    mkdir $base_dir/experiments/temp
    mkdir $base_dir/experiments/temp/in_dir
    mkdir $base_dir/experiments/temp/out_dir
    in_dir=$(echo $base_dir/experiments/temp/in_dir)
    out_dir=$(echo $base_dir/experiments/temp/out_dir)
    cp $in_path $in_dir/
else
    in_dir=$(echo $in_path)
    out_dir=$(echo $out_path)
fi

python $base_dir/automate_preprocessing.py $in_dir $base_dir/experiments/temp

mkdir $base_dir/experiments/temp/$model_name

python $base_dir/run_summarization.py --mode=decode --data_path=$base_dir/experiments/temp/finished_files/chunked/test_* --vocab_path=$base_dir/models/$vocab_name --log_root=$base_dir/models --exp_name=$model_name --max_enc_steps=$enc_step --max_dec_steps=$dec_step --coverage=1 --single_pass=1

mkdir -p $out_dir/text

mkdir -p $out_dir/attention_scores

mv $base_dir/models/$model_name/decode* $base_dir/experiments/temp/$model_name/decode

for i in $base_dir/experiments/temp/$model_name/decode/reference/*;
do
	j=$(echo $i | rev | cut -d '/' -f 1 | rev | cut -d '_' -f 1);
	target=$(cat $i | sed 's/ //g' | cut -d '.' -f 1);
	cat $base_dir'/experiments/temp/'$model_name'/decode/decoded/'$j'_decoded.txt' | sed -zE 's/[[:space:]]([,.?!])/\1/g' > $out_dir/text/$target'_'$enc_step'_'$dec_step.txt
	mv $base_dir'/experiments/temp/'$model_name'/decode/attention/'$j'_attn_vis_data.json' $out_dir/attention_scores/$target'_'$enc_step'_'$dec_step'.attn_vis_data.json'
done
echo "Summarization Successfully completed"
