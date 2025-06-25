import torch
from data import get_loaders
from utils import build_model, CustomLoss
import numpy as np
from metrics import Evaluator
from tqdm import tqdm
import wandb
from arg_utils import Args
import os


args = Args().parse_args(known_only=True)
torch.manual_seed(args.seed)

if args.use_wandb:
    wandb.init(project="xdocvqa_v2", config=args, dir="/data/users/msouibgui/")

# build the model
model, processor = build_model(args.model, task="vqa", args=args)

if args.encoder_frozen:
    print("Freezing the encoder")
    for param in model.pix2struct_model.encoder.parameters():
        param.requires_grad = False
if args.decoder_frozen:
    print("Freezing the decoder")
    if args.model == "donut":
        for param in model.donut_model.decoder.parameters():
            param.requires_grad = False
    else:
        for param in model.pix2struct_model.decoder.parameters():
            param.requires_grad = False

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!") 
    print(f"the visible devices are {os.environ['CUDA_VISIBLE_DEVICES']}")
    model = torch.nn.DataParallel(model)
    
model.to(device)

# get the dataloaders
train_dataloader, val_dataloader, _ = get_loaders(processor, args.data_path, args.batch_size)
epochs = args.num_epochs

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)) # T_max should be set according to the number of epochs

l2_loss = torch.nn.MSELoss() # Mean Squared Error Loss for the token interaction loss
custom_loss_fn = CustomLoss(model_type=args.model)


def train_one_epoch(epoch, data_loader, save_path):
    model.train()
    print("Epoch:", epoch)
    running_total_loss, running_loss_text_1, running_loss_text_2, running_loss_mask, running_l1_reg, running_continuity_reg, running_l2_mask_attn, running_loss_mask_colpali = 0, 0, 0, 0, 0, 0, 0, 0
    
    alpha_1 = args.alpha_1 # first pass text loss weight
    alpha_2 = args.alpha_2 # second pass sufficiency text loss weight (S)
    gamma = args.gamma # mask minimality loss weight (M)
    beta = args.beta # colpali interaction loss weight (I)
 
    
    for idx, batch in enumerate(data_loader):
        
        if args.model == "pix2struct":
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            pos_encoding = flattened_patches[:, :, :2]
            attention_mask = batch.pop("attention_mask").to(device)
            target_mask_imgs = batch.pop("target_mask_imgs").to(device)
            header_ratios = batch.pop("header_ratios").to(device)

            
            mask, outputs1, outputs2, attn, mask_out_img = model(flattened_patches=flattened_patches,
                                            attention_mask=attention_mask,
                                            labels=labels,
                                            header_ratios=header_ratios,
                                            used_features=args.used_features)
        elif args.model == "donut":
            pos_encoding = None
            labels = batch.pop("labels_donut").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            decoder_input_ids = batch.pop("decoder_input_ids").to(device)
            target_mask_imgs = batch.pop("target_mask_imgs").to(device)

            mask, outputs1, outputs2, mask_out_img= model(pixel_values, decoder_input_ids=decoder_input_ids[:, :-1], labels=labels[:, 1:])

        loss_mask, l1_reg, continuity_reg = custom_loss_fn.mask_loss(mask, pos_encoding=pos_encoding)
        
        loss_mask_colpali = l2_loss(mask_out_img, target_mask_imgs)
        loss_text_1 = outputs1.loss
        if args.use_distillation:
            kl_weight = 0.7
            kl_div_loss = custom_loss_fn.KD_loss(outputs1, outputs2)
            loss_text_2 = (1-kl_weight) * outputs2.loss + kl_weight * kl_div_loss * 1000
        else:
            loss_text_2 = outputs2.loss

        running_loss_text_1 += loss_text_1.item()
        running_loss_text_2 += loss_text_2.item()
        running_loss_mask += loss_mask.item()
        running_l1_reg += l1_reg.item()
        running_continuity_reg += continuity_reg.item()
        running_loss_mask_colpali += loss_mask_colpali.item()
        
        overall_loss = alpha_1 * loss_text_1 + alpha_2 * loss_text_2 + gamma * loss_mask + beta * loss_mask_colpali
        running_total_loss += overall_loss.item()

        show_every = 25 # log losses every 25 steps
        if idx % show_every == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{idx}/{len(data_loader)}] - "
                f"Loss Text 1: {running_loss_text_1 / show_every:.4f}, "
                f"Loss Text 2: {running_loss_text_2 / show_every:.4f}, "
                f"Loss Mask: {running_loss_mask / show_every:.4f}, "
                f"Loss M_l1: {running_l1_reg / show_every:.4f}, "
                f"Loss M_cont: {running_continuity_reg / show_every:.4f}, "
                f"Loss M_colpali: {running_loss_mask_colpali / show_every:.4f}, "
                f"Tot. Loss: {running_total_loss / show_every:.4f}")
            
            running_total_loss, running_loss_text_1, running_loss_text_2, running_loss_mask, running_l1_reg, running_continuity_reg, running_loss_mask_colpali = 0, 0, 0, 0, 0, 0, 0
            
        if idx % 100 ==0: # save model every 100 steps
            print("Saving model")
            torch.save(model.state_dict(), os.path.join(save_path,"last_model.pth"))
            print("Model saved")
        if args.use_wandb:
            wandb.log({"loss_text_1": loss_text_1.item(),
                        "loss_text_2": loss_text_2.item(),
                        "loss_mask": loss_mask.item(),
                        "colpali_loss": loss_mask_colpali.item(),
                        "total_loss": overall_loss.item()})
        overall_loss.backward()
        optimizer.step()
        if idx >=5000 and epoch > 1:
            scheduler.step()

def eval_docvqa(model):
    model.eval()
    anls = 0
    acc = 0
    batch_idx = 0
    for batch in tqdm(val_dataloader):
        gt_answers = batch.pop("gt_answers")
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)

        with torch.no_grad():
            predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask,  max_length=100)
        pred_answers = processor.batch_decode(predictions, skip_special_tokens=True)
        metric = evaluator.get_metrics(gt_answers, pred_answers)

        batch_acc = np.mean(metric['accuracy'])
        batch_anls = np.mean(metric['anls'])
        acc += batch_acc
        anls += batch_anls
        batch_idx += 1

    if args.use_wandb:
        wandb.log({"acc_doqvqa": acc / batch_idx, "anls_doqvqa": anls / batch_idx})
    return acc / batch_idx, anls / batch_idx

def eval_masked_doc(model):
    model.eval()
    anls = 0
    acc = 0
    batch_idx = 0
    for batch in tqdm(val_dataloader):
        gt_answers = batch.pop("gt_answers")
        flattened_patches = batch.pop("flattened_patches").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        labels = batch.pop("labels").to(device)
        header_ratios = batch.pop("header_ratios").to(device)
        with torch.no_grad():
            
            if args.used_features == "enc_dec_attn":
                pos_encoding = flattened_patches[:, :, :2]
                p1 = int(max(list(flattened_patches[:,:,0])[0]))
                p2 = int(max(list(flattened_patches[:,:,1])[0]))
                patches = p1*p2

                encoder_output = model.pix2struct_model.encoder(flattened_patches, output_hidden_states=True)
                features = encoder_output['last_hidden_state']
                decoder_input_ids = model.pix2struct_model._shift_right(labels)
                decoder_attention_mask = decoder_input_ids.ne(model.pix2struct_model.config.pad_token_id).float()
                decoder_attention_mask[:, 0] = 1

                decoder_output = model.pix2struct_model.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    encoder_attention_mask=attention_mask,
                    labels=labels,
                    encoder_hidden_states = encoder_output[0],
                    output_hidden_states=True,
                    output_attentions=True
                )

                cross_attention_11 = decoder_output["cross_attentions"][11]
                
                sequence_logits = decoder_output['logits']
                sequence_classes = torch.argmax(sequence_logits, 2)
                lenghts = [len(sq[sq!= 0])-1 for sq in sequence_classes]
                all_heads_attn_11 = torch.stack([torch.sum(torch.sum(cross_attention_11[k,:,:i,:args.max_patches], 1), 0).unsqueeze(-1) for k, i in enumerate(lenghts)])
                attn_rep = all_heads_attn_11.repeat(1, 1, 768)
                feat_attn = torch.cat((pos_encoding, features, attn_rep), 2)
                mask_out = model.mask_head_deocder(feat_attn)
                mask_out = model.ste(mask_out)

            num_tokens = flattened_patches.size(1)
            batch_header_rows = torch.ceil(pos_encoding[:,:,0].max(1)[0] * header_ratios).int()
            batch_header_tokens = batch_header_rows * pos_encoding[:,:,0].max(1)[0] 
            batch_tokens = torch.arange(num_tokens, device=flattened_patches.device)

            mask_2 = batch_tokens.unsqueeze(0) < batch_header_tokens.unsqueeze(1)

            mask_out[mask_2] = 1.0

            flattened_patches_masked = mask_out * flattened_patches
            predictions = model.pix2struct_model.generate(flattened_patches=flattened_patches_masked, attention_mask=attention_mask)
        
        pred_answers = processor.batch_decode(predictions, skip_special_tokens=True)
        metric = evaluator.get_metrics(gt_answers, pred_answers)
        batch_acc = np.mean(metric['accuracy'])
        batch_anls = np.mean(metric['anls'])
        acc += batch_acc
        anls += batch_anls
        batch_idx += 1

    if args.use_wandb:
        wandb.log({"acc_masked_doqvqa": acc / batch_idx, "anls_masked_doqvqa": anls / batch_idx})
    return acc / batch_idx, anls / batch_idx

if __name__ == "__main__":
    evaluator = Evaluator(case_sensitive=False)
    best_anls = 0
    print("Training for ", epochs, " epochs")
    save_path = os.path.join(args.save_path, f"{args.model}_{args.dataset}", args.experiment_name)
    
    if args.ckpt_path is not None:
        print("Loading model from ", args.ckpt_path)
        load_path = os.path.join(args.save_path, args.ckpt_path)
        model.load_state_dict(torch.load(os.path.join(load_path, "last_model.pth")))
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for epoch in range(epochs):
        print("Training epoch: ", epoch, " /  Total epochs: ", epochs)
        train_one_epoch(epoch, train_dataloader, save_path)
        print("Evaluation docvqa epoch: ", epoch)
        acc, anls = eval_docvqa(model.pix2struct_model)
        print("acc: ", acc, "\n anls: ", anls)
        print("Evaluation masked docvqa epoch: ", epoch)
        acc, anls = eval_masked_doc(model)
        print("acc: ", acc, "\n anls: ", anls)
        
        # save checkpoint of last and best model
        if anls > best_anls:
            best_anls = anls
            print("Saving model")
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        
        print("Saving model for epoch: ", epoch)
        torch.save(model.state_dict(), os.path.join(save_path, "model_epoch_"+str(epoch)+".pth"))