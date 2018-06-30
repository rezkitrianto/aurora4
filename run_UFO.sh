. ./path.sh

# If you have cluster of machines running GridEngine you may want to
# change the train and decode commands in the file below
. ./cmd.sh


Step1_DoDataPrep=0         # do data preparation with kaldi format
Step2_DoDictPrep=0         # do dict preparation
Step3_DoLangPrep=0         # do phone set hmm topology preparation
Step4_DoLM2FST=0           # do Arpa LM to FST conversion
Step5_DoFeatureExt=0      # do feature process
Step6_DoMonoTriTrain=1    # do monophone acoustic model training
Step7e_DoLDAMLLTEval=1
Step12_DoDNN_fBank=1
Step13_DoDNNDecode_fBank=1
if [ $Step1_DoDataPrep -eq 1 ]; then
   	aurora4=/home/msl/hdd1T/4A
	#wsj0=/share/WSJ0
	#local/aurora4_data_prep.sh $aurora4A $wsj0 $aurora4B
	local/aurora4_data_prep2.sh $aurora4 $wsj0
fi

# step 2  prepare dict

if [ $Step2_DoDictPrep -eq 1 ]; then
	echo -e "\nprepare dict on progress ...\n";
	local/wsj_prepare_dict.sh || exit 1;
fi

# step 3 prepare lang_tmp folder, including
#  3.1  generate extended phone list with boundary info,
#       which is necessary to locate the word boundary from decoded
#        hypothesis
#  3.2  create some extra question list from phone list
#  3.3  add disambiguate symbols to lexicon, to let
#       optimization of Finite State Transducer decoding network
#       feasible
#  3.4  create word/phone list table
#  3.5  define HMM prototype topology
#  3.6  make FST for lexicon
if [ $Step3_DoLangPrep -eq 1 ] ; then
	utils/prepare_lang.sh data/local/dict "<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;
fi

# step 4 prepare language model and corresponding FST
if [ $Step4_DoLM2FST -eq 1 ]; then
	passit=true
	 echo -e "\nlanguage model preparation on progress\n";
  	if passit; then
    		local/wsj_train_lms.sh
  	fi
	# do arpa lm to FST conversion
  	utils/format_lm.sh
  	echo -e "\nlanguage model preparation failed\n";
fi

if [ $Step5_DoFeatureExt -eq 1 ]; then
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc

for x in train_si84_clean train_si84_multi test_eval92_01 test_eval92_02 test_eval92_03 test_eval92_04 test_eval92_05 test_eval92_06 test_eval92_07 test_eval92_08 test_eval92_09 test_eval92_10 test_eval92_11 test_eval92_12 test_eval92_13 test_eval92_14; do
	steps/make_mfcc.sh  --nj 10 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
	done

# make fbank features
fbankdir=fbank
for x in train_si84_multi test_eval92_01 test_eval92_02 test_eval92_03 test_eval92_04 test_eval92_05 test_eval92_06 test_eval92_07 test_eval92_08 test_eval92_09 test_eval92_10 test_eval92_11 test_eval92_12 test_eval92_13 test_eval92_14; do
	steps/make_fbank.sh --nj 10 data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
	steps/compute_cmvn_stats.sh data-fbank/$x exp/make_fbank/$x $fbankdir || exit 1;
	done
fi

if [ $Step6_DoMonoTriTrain -eq 1 ]; then

	#steps/train_mono.sh --boost-silence 1.25 --nj 10 data/train_si84_multi data/lang exp/mono0a_multi || exit 1;

	#steps/align_si.sh --boost-silence 1.25 --nj 10 \
	#	data/train_si84_multi data/lang exp/mono0a_multi exp/mono0a_multi_ali || exit 1;
	#steps/train_deltas.sh --boost-silence 1.25 2000 10000 \
	#	 data/train_si84_multi data/lang exp/mono0a_multi_ali exp/tri1_multi || exit 1;


	#steps/align_si.sh --nj 10 data/train_si84_multi data/lang exp/tri1_multi exp/tri1_multi_ali_si84 || exit 1;
	#steps/train_deltas.sh --boost-silence 1.25 2000 10000 \
	#	 data/train_si84_multi data/lang exp/mono0a_multi_ali exp/tri1_multi || exit 1;
	
	#steps/train_deltas.sh  \
  	#	2500 15000 data/train_si84_multi data/lang exp/tri1_multi_ali_si84 exp/tri2a_multi || exit 1;

	steps/train_lda_mllt.sh --splice-opts "--left-context=3 --right-context=3" \
   		2500 15000 data/train_si84_multi data/lang exp/tri1_multi_ali_si84 exp/tri2b_multi || exit 1;

	utils/mkgraph.sh data/lang_test_tgpr_5k exp/tri2b_multi exp/tri2b_multi/graph_tgpr_5k || exit 1;

fi

if [ $Step7e_DoLDAMLLTEval -eq 1 ]; then
  	for x in $(seq -f "%02g" 01 14); do
        	steps/decode.sh  --nj 7 --cmd "$decode_cmd" \
			exp/tri2b_multi/graph_tgpr_5k data/test_eval92_${x} exp/tri2b_multi/decode_tgpr_5k_eval92_${x} || exit 1;
  	done

	for x in $(seq -f "%02g" 01 14); do
  		less exp/tri2b_multi/decode_tgpr_5k_eval92_${x}/wer_*  | utils/best_wer.sh > gmmresult.log
	done
fi

dir=exp/tri3a_dnn_multi_cmvn
ali=exp/tri2b_multi_ali_si84
feature_transform=exp/tri3a_dnn_multi_pretrain_cmvn/final.feature_transform
pretridir=exp/tri3a_dnn_multi_pretrain_cmvn
tri2b=exp/tri2b_multi

if [ $Step12_DoDNN_fBank -eq 1 ]; then
	
	steps/align_si.sh  --nj 10 --use-graphs true \
		data/train_si84_multi data/lang exp/tri2b_multi $ali  || exit 1;

	echo "111"

	###================fix by Ting-Yao=============#

	$cuda_cmd $pretridir/_pretrain_dbn.log \
		steps/nnet/pretrain_dbn.sh --nn-depth 5 --rbm-iter 3 data-fbank/train_si84_multi $pretridir

	echo "222"
	
	$cuda_cmd $dir/_train_nnet.log steps/nnet/train.sh \
		--feature-transform $feature_transform --dbn $pretridir/5.dbn --hid-layers 0 --learn-rate 0.008 \
 		data-fbank/train_si84_multi data-fbank/train_si84_multi data/lang $ali $ali $dir || exit 1;

	echo "333"
	
	utils/mkgraph.sh data/lang_test_tgpr_5k $dir $dir/graph_tgpr_5k || exit 1;

fi

if [ $Step13_DoDNNDecode_fBank -eq 1 ]; then
     
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_01 $dir/decode_tgpr_5k_test_eval92_01 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_02 $dir/decode_tgpr_5k_test_eval92_02 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_03 $dir/decode_tgpr_5k_test_eval92_03 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_04 $dir/decode_tgpr_5k_test_eval92_04 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_05 $dir/decode_tgpr_5k_test_eval92_05 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_06 $dir/decode_tgpr_5k_test_eval92_06 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_07 $dir/decode_tgpr_5k_test_eval92_07 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_08 $dir/decode_tgpr_5k_test_eval92_08 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_09 $dir/decode_tgpr_5k_test_eval92_09 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_10 $dir/decode_tgpr_5k_test_eval92_10 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_11 $dir/decode_tgpr_5k_test_eval92_11 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_12 $dir/decode_tgpr_5k_test_eval92_12 || exit 1
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_13 $dir/decode_tgpr_5k_test_eval92_13 || exit 1        
        steps/nnet/decode.sh --nj 4 --acwt 0.10 --config conf/decode_dnn.config $dir/graph_tgpr_5k \
                data-fbank/test_eval92_14 $dir/decode_tgpr_5k_test_eval92_14 || exit 1    

        for x in $(seq -f "%02g" 01 14); do
                less $dir/decode_tgpr_5k_test_eval92_${x}/wer_*  | utils/best_wer.sh >> $dir/best_${x}
        done
	
	cat $dir/best_*
fi
