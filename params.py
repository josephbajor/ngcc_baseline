import argparse


def get_params():
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument("--run_name", type=str, default="NGCC_DOA", help="name of run")
    parser.add_argument("--load", type=str, default=None, help="run name to load")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/jbajor/workhorse3/ngcc_baseline/experiments/",
        help="save path for model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of training epochs",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--nworkers", type=int, default=1, help="number of workers")
    parser.add_argument("--logger", type=str, default=None, help="logger option")
    parser.add_argument("--lr", type=float, default=4e-4, help="base learning rate")
    parser.add_argument(
        "--val_every_n_epochs",
        type=int,
        default=1,
        help="check validation set every n epochs",
    )
    parser.add_argument("--gpus", type=int, default=1, help="train on n gpus")
    parser.add_argument(
        "--max_epochs", type=int, default=300, help="max number of epochs"
    )
    parser.add_argument(
        "--overfit_batches", type=float, default=0.0, help="overfit batches"
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="debug flag"
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val/loss",
        help="metric to monitor for callbacks",
    )
    parser.add_argument("--mode", type=str, default="min", help="min or max")
    parser.add_argument(
        "--patience", type=int, default=20, help="patience for callbacks"
    )
    parser.add_argument(
        "--min_delta", type=int, default=0, help="tolerance for callbacks"
    )
    parser.add_argument("--seed", type=int, default=18792, help="random seed")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument(
        "--schedule_lr",
        default=False,
        action="store_true",
        help="use lr scheduler (reduce on plateau)",
    )

    # Model arguments
    parser.add_argument("--model", type=str, default="general", help="model type")
    parser.add_argument(
        "--bi", default=False, action="store_true", help="bidirectional rnn"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.15, help="dropout probability"
    )
    parser.add_argument(
        "--rnn_layers", type=int, default=4, help="number of rnn layers"
    )
    parser.add_argument("--up_factor", type=int, default=1, help="upsample factor")
    parser.add_argument(
        "--win_len", type=float, default=0.02, help="window length for stft"
    )
    parser.add_argument(
        "--hop_frac",
        type=float,
        default=0.5,
        help="hop length as fraction of window length",
    )
    parser.add_argument("--kernel_size", type=int, default=9, help="cnn kernel size")
    parser.add_argument("--nchan", type=int, default=1, help="number of cnn channels")
    parser.add_argument("--stride", type=int, default=1, help="cnn stride")
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="rnn output/dnn input dimensions"
    )
    parser.add_argument(
        "--model_max_lag",
        type=float,
        default=None,
        help="maximum lag range seen by DOA model (seconds)",
    )
    parser.add_argument(
        "--frontend_weights",
        type=str,
        default=None,
        help="path to pretrained frontend weights; `None` gives random initialization",
    )
    parser.add_argument(
        "--backend_weights",
        type=str,
        default=None,
        help="path to pretrained backend weights; `None` gives random initialization",
    )
    parser.add_argument(
        "--freeze", default=False, action="store_true", help="freeze learnable frontend"
    )
    parser.add_argument(
        "--freeze_back", default=False, action="store_true", help="freeze DOANet"
    )
    parser.add_argument(
        "--cc_win",
        type=str,
        default="rect",
        help="window applied to input feature; `rect`, `ham`, or `hann`",
    )
    parser.add_argument(
        "--lwin",
        default=False,
        action="store_true",
        help="make input feature window learnable",
    )
    parser.add_argument(
        "--perfect_start", default=False, action="store_true", help="start LCC as GCC"
    )
    parser.add_argument(
        "--cnn_fbank", default=False, action="store_true", help="use learned filterbank"
    )
    parser.add_argument(
        "--cnn_layers", type=int, default=2, help="layers of cnn filterbank"
    )
    parser.add_argument(
        "--backend_type",
        type=str,
        default="mlp",
        choices=["mlp", "crdnn", "rnn", "cnn", "max"],
        help="Type of backend used in the model that computes DoA",
    )
    parser.add_argument(
        "--temporal_pool",
        type=str,
        default="mean",
        choices=["mean", "selfattn"],
        help="Method to perform temporal pooling- currently accepts mean and selfattn",
    )
    parser.add_argument(
        "--resample",
        type=int,
        default=16000,
        help="Resample audio on load to specified sample rate",
    )

    # Data
    parser.add_argument(
        "--speech_root",
        type=str,
        default="/home/marklind/workhorse1/vbd",
        help="speech directory",
    )
    parser.add_argument(
        "--rir_root",
        type=str,
        default="/home/marklind/workhorse1/tuburo_correct_xlarge",
        help="rir directory",
    )
    parser.add_argument(
        "--noise_root",
        type=str,
        default="/home/marklind/cmu-se-lightning-hydra/data/vbd",
        help="noise directory",
    )
    parser.add_argument(
        "--noise_type", type=str, default="none", help="add noise to dataset"
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.0,
        help="probability of adding noise during training",
    )
    parser.add_argument(
        "--keyring_root",
        type=str,
        default="/home/jbajor/keyring/",
        help="keyring directory",
    )
    parser.add_argument(
        "--white_noise",
        default=False,
        action="store_true",
        help="use white noise in place of vbd noise",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="white noise snr; `None` will mimic vbd snr",
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="sample rate")
    parser.add_argument(
        "--fixed_duration", type=float, default=1.0, help="duration of training samples"
    )
    parser.add_argument(
        "--middle", type=bool, default=True, help="take samples from middle of signal"
    )
    parser.add_argument(
        "--simple_input",
        default=False,
        action="store_true",
        help="simplifies input to just delays to left or right",
    )
    parser.add_argument(
        "--n_angles",
        type=int,
        default=3,
        help="number of different angles for simple input",
    )
    parser.add_argument(
        "--n_look_angles",
        type=int,
        default=36,
        help="number of different look angles for feature",
    )
    parser.add_argument(
        "--binary",
        default=False,
        action="store_true",
        help="simplifies problem to binary classification",
    )
    parser.add_argument(
        "--sa", default=False, action="store_true", help="flag for SpecAugment"
    )
    parser.add_argument(
        "--sa_time_width",
        type=int,
        default=20,
        help="width of time mask for SpecAugment",
    )
    parser.add_argument(
        "--sa_freq_width",
        type=int,
        default=20,
        help="width of frequency mask for SpecAugment",
    )
    parser.add_argument(
        "--n_sa_time", type=int, default=2, help="number time masks for SpecAugment"
    )
    parser.add_argument(
        "--n_sa_freq", type=int, default=2, help="number time masks for SpecAugment"
    )
    parser.add_argument(
        "--degrees",
        default=True,
        action="store_true",
        help="flag to show error in degrees; otherwise radians",
    )
    parser.add_argument(
        "--phat",
        default=False,
        action="store_true",
        help="flag for GCC PHAse Transform",
    )
    parser.add_argument(
        "--input_feat",
        type=str,
        default="math",
        help="math or learnable input features; `math` or `learn`",
    )
    parser.add_argument(
        "--window", default=False, action="store_true", help="window the stana function"
    )
    parser.add_argument(
        "--norm_cc",
        default=False,
        action="store_true",
        help="normalize time-domain cc function",
    )

    # Frontend experiment flags
    parser.add_argument(
        "--train_sample_type",
        type=str,
        default="speech",
        help="training data sample type; `wn`, `tone`, `speech`, or `all`",
    )
    parser.add_argument(
        "--val_sample_type",
        type=str,
        default="speech",
        help="validation data sample type; `wn`, `tone`, `speech`, or `all`",
    )
    parser.add_argument(
        "--sep_len", type=float, default=0.5, help="simulated mic separation"
    )
    parser.add_argument(
        "--sample_len", type=int, default=1.0, help="duration of one sample"
    )
    parser.add_argument(
        "--n_train_samples",
        type=int,
        default=1000,
        help="number of samples to use of the train set",
    )
    parser.add_argument(
        "--n_val_samples",
        type=int,
        default=100,
        help="number of samples to use of the train set",
    )
    parser.add_argument(
        "--target_win",
        type=str,
        default="rect",
        help="window applied to input feature; `rect`, `ham`, or `hann`",
    )
    parser.add_argument(
        "--target_phat", default=False, action="store_true", help="apply PHAT to target"
    )
    parser.add_argument(
    "--exp_name", type=str, default="ngccphat", help="Name of the experiment"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Set to true in order to evaluate the model across a range of SNRs and T60s",
    )
    parser.add_argument(
        "--eval_nogen",
        action="store_true",
        default=True,
        help="Flag to enable loading of local data for evaluation instead of generating",
    )

    return parser.parse_args()
