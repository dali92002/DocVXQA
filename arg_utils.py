from tap import Tap

class Args(Tap):
    dataset_names = ["SPDocVQA", "infographics_vqa", "pfl_docvqa"]  # list of dataset names
    model_names = ["donut", "pix2struct"]  # list of model names
    model: str = model_names[1]  # "donut" or "pix2struct" # model name (donut or pix2struct)
    dataset: str = dataset_names[0] # "SPDocVQA"# "infographics_vqa"  #"pfl_docvqa"
    batch_size: int = 5  
    max_length: int = 40  # maximum length of the labels text (tokenized)
    num_epochs: int = 100  # number of epochs
    lr: float = 1e-7 # learning rate final S, M, I losses
    seed: int = 42 # seed
    alpha_1 = 2 # first pass text loss weight
    alpha_2 = 1 # second pass sufficiency text loss weight (S)
    gamma = 0.5 # mask minimality loss weight (M)
    beta = 5 # colpali interaction loss weight (I)
    max_patches: int = 2048  # maximum number of patches for pix2struct
    use_proc_data: bool = False  # use processed data, if exist, for faster training
    if dataset == "SPDocVQA":
        # use_proc_data: bool = False  # use processed data
        data_path: str = "/data-net/125/users/msouibgui/datasets/SPDocVQA/"  # path to the data
        save_path: str = "/data/users/msouibgui/ckpts/XdocVQA_v2/"  # path to save the model
        colpali_path: str = "/data/users/msouibgui/datasets/SPDocVQA/colpali_raw/"#"/data/users/msouibgui/datasets/SPDocVQA/colpali/" # path to the colpali generated masks
        base_pix2struct_path: str = "/data/users/msouibgui/ckpts/XdocVQA/XDocVQA_1_model/best_model_p2x_original.pth"
        # eval_colpali: bool = False  # evaluate the colpali model
    elif dataset == "infographics_vqa":
        data_path: str = f"/data/users/msouibgui/datasets/{dataset}/"  # path to the dataset
        save_path: str = f"/data/users/msouibgui/ckpts/XdocVQA_v2/{dataset}/"  # path to save the model
        colpali_path: str = f"/data/users/msouibgui/datasets/{dataset}/colpali_raw/"#"/data/users/msouibgui/datasets/SPDocVQA/colpali/" # path to the colpali generated masks
    elif dataset == "pfl_docvqa":
        data_path: str = "/data-net/125/shared/PFL-DocVQA/imdb/1.0centralized/"
        save_path: str = "/data/users/msouibgui/ckpts/XdocVQA_v2/pfl_docvqa/"  # path to save the model
        colpali_path: str = f"/data/users/msouibgui/datasets/{dataset}/colpali_raw/"#"/data/users/msouibgui/datasets/SPDocVQA/colpali/" # path to the colpali generated masks
        base_pix2struct_path: str = "/data/users/msouibgui/ckpts/XdocVQA_v2/pfl_docvqa/XDocVQA_v2_colpali_raw/best_model_p2x_original.pth"  # path to the base pix2struct model
    
    eval_colpali: bool = False  # evaluate only the colpali model
    use_wandb: bool = True  # use wandb for logging
    experiment_list = ["XDocVQA_2_models", "XDocVQA_1_model", "XDocVQA_1_model_teacher"]  # list of experiments
    experiment_name: str = f"{model}_{dataset}" # name of the experiment
    use_distillation: bool = False  # use distillation loss
    use_bce: bool = False  # use BCE loss in the mask loss
    lambda_reg: float = 0.3  # regularization parameter for the mask loss
    debug_mode: bool = False  # debug mode
    project_mode: str = "final"  # project milestone
    used_features: str = "enc_dec_attn"  # feature location (encoder, decoder)
    encoder_frozen: bool = False  # freeze the encoder
    decoder_frozen: bool = False  # freeze the decoder
    
    use_attentions: bool = True  # use attentions
    colpali_mode: bool = False  # use colpali mode
    ckpt_path: str = None  # path to the checkpoint
    
    start: int = 0  # start iteration
    end: int = 10000  # end iteration   