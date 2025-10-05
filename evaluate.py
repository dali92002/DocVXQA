import torch
from data import get_loaders
from models.pix2struct import get_pix2struct_model
from models.model import XDocVQA
import numpy as np
from metrics import Evaluator
from tqdm import tqdm
import wandb
from arg_utils import Args
import os
from utils import post_process_mask
from einops import rearrange
import json

args = Args().parse_args(known_only=True)
# set args to wandb
if args.use_wandb:
    wandb.init(config=args)

# get the pix2struct model and backbone
base_pix2struct, processor = get_pix2struct_model(task="vqa")
pix2struct, processor = get_pix2struct_model(task="vqa")
model = XDocVQA(pix2struct_model=pix2struct, max_patches=args.max_patches)


device = "cuda" if torch.cuda.is_available() else "cpu"
base_pix2struct.to(device)
model.to(device)

# get the dataloaders
_, val_dataloader, _ = get_loaders(processor, args.data_path, args.batch_size)

def find_bg(features, mask_out):
    features = torch.round(features[:,:, 2:] * 10000) / 10000
    for i in range(features.size(1)):
        if torch.var(features[:,i]) <= 0.01:
            mask_out[:,i] = 0.0
    return mask_out



def eval_masked_doc(model, base_pix2struct=None, post_process=False, k_postprocess=1, th=0.5):
    model.eval()
    anls = 0
    acc = 0
    batch_idx = 0
    results = {"acc": [], "anls": [], "pred_answers": [], "mask_ratio": []}
    for batch in tqdm(val_dataloader):
        gt_answers = batch.pop("gt_answers")
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        labels = batch.pop("labels").to(device)
        header_ratios = batch.pop("header_ratios").to(device)
        # answer_types = batch.pop("answer_types")
        with torch.no_grad():
            if True:
                encoder_output = model.pix2struct_model.encoder(flattened_patches, output_hidden_states=True)
                features = encoder_output['last_hidden_state']
                decoder_input_ids = model.pix2struct_model._shift_right(labels)
                decoder_attention_mask = decoder_input_ids.ne(model.pix2struct_model.config.pad_token_id).float()
                decoder_attention_mask[:, 0] = 1
                pos_encoding = flattened_patches[:,:,:2]

                decoder_output = model.pix2struct_model.decoder(
                    input_ids=decoder_input_ids,
                    attention_m884ask=decoder_attention_mask,
                    encoder_attention_mask=attention_mask,
                    labels=labels,
                    encoder_hidden_states = encoder_output[0],
                    output_hidden_states=True,
                    output_attentions=True
                )


                cross_attention_11 = decoder_output["cross_attentions"][11]

                attn_11 = torch.sum(cross_attention_11[:,:,0,:args.max_patches], 1)
                p1 = [int(max(l)) for l in list(flattened_patches[:,:,0].cpu().numpy())]
                p2 = [int(max(l)) for l in list(flattened_patches[:,:,1].cpu().numpy())]
                patches = np.array(p1)*np.array(p2)
                att_imgs = [flattened_patches[i,:patches[i],2:].clone() for i in range(len(patches))]

                # att_img = flattened_patches[:,:patches,2:].clone()
                for i in range(len(patches)):
                    for j in range(patches[i]):
                        att_imgs[i][j,:] = attn_11[i, j]
                    att_imgs[i] = rearrange(att_imgs[i].unsqueeze(0), 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                                p1 = 16, p2 = 16,  h=p1[i]).cpu().squeeze(0)
                
                # len of outputtes sequences
                sequence_logits = decoder_output['logits']
                sequence_classes = torch.argmax(sequence_logits, 2)
                lenghts = [len(sq[sq!= 0])-1 for sq in sequence_classes]
                all_heads_attn_11 = torch.stack([torch.sum(torch.sum(cross_attention_11[k,:,:i,:args.max_patches], 1), 0).unsqueeze(-1) for k, i in enumerate(lenghts)])
                all_heads_attn_11 = all_heads_attn_11.repeat(1, 1, 768)
                feat_attn = torch.cat((pos_encoding, features, all_heads_attn_11), 2)
                mask_out_raw = model.mask_head_deocder(feat_attn)

                # mask_out = model.ste(mask_out)
                mask_out = []
                for m_o in mask_out_raw:
                    m_o = m_o
                    mask_out.append((m_o > torch.quantile(m_o, th).item()).float())
                mask_out = torch.stack(mask_out)
                # mask_out = (mask_out > torch.quantile(mask_out, th).item()).float()
                
                mask_out = find_bg(flattened_patches, mask_out)
            
            num_tokens = flattened_patches.size(1)
            batch_header_rows = torch.ceil(pos_encoding[:,:,0].max(1)[0] * header_ratios).int()
            batch_header_tokens = batch_header_rows * pos_encoding[:,:,1].max(1)[0] 
            batch_tokens = torch.arange(num_tokens, device=flattened_patches.device)
            
            img_header = batch_tokens.unsqueeze(0) < batch_header_tokens.unsqueeze(1)
            
            if post_process:
                mask_out = post_process_mask(mask_out, att_imgs, p1, p2, k_postprocess, img_header)
            mask_out[img_header] = 0.0
            mask_ratio = mask_out.sum(1).squeeze() / (num_tokens - (img_header).float().sum(1)) 
            mask_out[img_header] = 1.0

            flattened_patches_masked = mask_out * flattened_patches
            if base_pix2struct is not None:
                predictions = base_pix2struct.generate(flattened_patches=flattened_patches_masked, attention_mask=attention_mask)
            else:
                predictions = model.pix2struct_model.generate(flattened_patches=flattened_patches_masked, attention_mask=attention_mask,  max_length= 100)
        pred_answers = processor.batch_decode(predictions, skip_special_tokens=True)
        from utils import patches_to_img
        patches_to_img(flattened_patches[0].unsqueeze(0), flattened_patches_masked[0].unsqueeze(0), mean =[0.8937789027534004, 0.8937789027534004, 0.8937789027534004], std= [0.27155470203047904, 0.27155470203047904, 0.27155470203047904]).save("test.png")

        metric = evaluator.get_metrics(gt_answers, pred_answers)


        
        results["acc"] += metric['accuracy']
        results["anls"] += metric['anls']
        results["pred_answers"] += pred_answers
        results["mask_ratio"] += mask_ratio.tolist()
        
        batch_acc = np.mean(metric['accuracy'])
        batch_anls = np.mean(metric['anls'])
        acc += batch_acc
        anls += batch_anls
        batch_idx += 1
    return acc / batch_idx, anls / batch_idx, results


if __name__ == "__main__":
    threshhold = 0.8
    k = 3
    model_path = "./weights/docvxqa_sp.pth"
    print("Evaluation docvqa k = ", k)    
    evaluator = Evaluator(case_sensitive=False)
    best_anls = 0
    
    model.load_state_dict(torch.load(model_path))
    all_results = {}

    print("Evaluation masked docvqa post process k= ", k, " th = ", threshhold)
    acc, anls, results = eval_masked_doc(model, post_process=True, k_postprocess=k, th = threshhold)
    
    pixel_ratio  = np.array(results["mask_ratio"])
    nan_mask = np.isnan(pixel_ratio)
    pixel_ratio = pixel_ratio[~nan_mask]
    pixel_ratio = pixel_ratio.mean()
    
    print("k= ", k, " th = ", threshhold, "\n acc: ", acc, "\n anls: ", anls, "\n pixel_ratio: ", pixel_ratio)
    all_results["masked_docvqa_post_process"] = results
    if args.use_wandb:
        wandb.log({"acc_masked_doqvqa process": acc, "anls_masked_doqvqa process": anls, "pixel_ratio": pixel_ratio})

    all_results["masked_docvqa_original_p2s_m"] = results

    # save results as json per threshold
    saving_dir = os.path.join("results", f"{args.model}_{args.dataset}_", "eval_results")
    os.makedirs(saving_dir, exist_ok=True)
    with open(os.path.join(saving_dir,"results_k_"+str(k)+"_th_"+str(threshhold)+".json"), "w") as f:
        json.dump(all_results, f, indent=4)