while getopts ":i:o:p:e:d:" opt; do
  case $opt in
    i) in_dir="$OPTARG"
    ;;
    o) out_dir="$OPTARG"
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

in_dir=$(echo $in_dir | sed 's/\/$//g')
out_dir=$(echo $out_dir | sed 's/\/$//g')
base_dir=$(echo $base_dir | sed 's/\/$//g')
model_name='summary-model'
vocab_name='summary-vocab'
export CLASSPATH=$base_dir/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar


# Remove the temp folder (sentence_per_line, transcripts_tokenized, finished_files)
rm -r $base_dir/experiments/temp

python $base_dir/automate_preprocessing.py $in_dir

mkdir $base_dir/experiments/temp/$model_name

python $base_dir/run_summarization.py --mode=decode --data_path=$base_dir/experiments/temp/finished_files/chunked/test_* --vocab_path=$base_dir/models/$vocab_name --log_root=$base_dir/experiments/temp --exp_name=$model_name --max_enc_steps=$enc_step --max_dec_steps=$dec_step --coverage=1 --single_pass=1

mkdir -p $out_dir/text

mkdir -p $out_dir/attention_scores

mv $base_dir/experiments/temp/$model_name/decode* $base_dir/experiments/temp/$model_name/decode

for i in $base_dir/experiments/temp/$model_name/decode/reference/*;
do
	j=$(echo $i | rev | cut -d '/' -f 1 | rev | cut -d '_' -f 1);
	target=$(cat $i | sed 's/ //g' | cut -d '.' -f 1);
	cat $base_dir'/models/'$model_name'/decode/decoded/'$j'_decoded.txt' | sed -zE 's/[[:space:]]([,.?!])/\1/g' > $out_dir/text/$target'_'$enc_step'_'$dec_step.txt
	mv $base_dir'/experiments/temp/'$model_name'/decode/attention/'$j'_attn_vis_data.json' $out_dir/attention_scores/$target'_'$enc_step'_'$dec_step'.attn_vis_data.json'
done
echo "Summarization Successfully completed"