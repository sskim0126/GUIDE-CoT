import argparse
from utils.config import get_exp_config, print_arguments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/config.json", type=str, help="Config file path.")
    parser.add_argument('--dataset', default="eth", type=str, help="Dataset name.", choices=["eth", "hotel", "univ", "zara1", "zara2"])
    parser.add_argument('--tag', default="LLM", type=str, help="Personal tag for the model.")
    parser.add_argument('--test', default=False, action='store_true', help="Evaluation mode.")
    parser.add_argument('--eval_mode', default='g2p', type=str, help="Evaluation mode.", \
                        choices=['g2p'])
    parser.add_argument('--phase', default=None, type=str, help="Phase of the model.", choices=[None, 'train', 'val', 'test'])
    
    args = parser.parse_args()

    print("===== Arguments =====")
    print_arguments(vars(args))

    print("===== Configs =====")
    cfg = get_exp_config(args.cfg)
    print_arguments(cfg)

    # Update configs
    cfg.dataset_name = args.dataset

    cfg.phase = args.phase

    if not args.test:
        # Training phase
        cfg.checkpoint_name = args.tag + '-' + cfg.train_dataset_type
        
        from model.trainval_accelerator import *
        trainval(cfg)

    else:
        # Evaluation phase
        cfg.checkpoint_name = args.tag
        
        if args.eval_mode == 'g2p':
            from model.eval_g2p_accelerator import *
        else:
            raise ValueError(f"Unknown evaluation mode: {args.eval_mode}")
        
        test(cfg)
