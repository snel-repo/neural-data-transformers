SYNTH_PATH=/snel/share/data/chaotic_rnn_data/data_no_inputs


python generate_chaotic_rnn_data.py --save_dir=$SYNTH_PATH --datafile_name=chaotic_rnn_no_inputs --synth_data_seed=5 --T=1.0 --C=130 --N=50 --S=50 --train_percentage=0.8 --nreplications=10 --g=1.5 --x0_std=1.0 --tau=0.025 --dt=0.01 --ninputs=0 --input_magnitude_list='' --max_firing_rate=30.0 --noise_type='poisson'