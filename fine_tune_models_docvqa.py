import torch
from data import get_loaders
from models.pix2struct import get_pix2struct_model
from models.donut import get_donut_model
import numpy as np
from metrics import Evaluator
from tqdm import tqdm
import wandb
from arg_utils import Args
import os

args = Args().parse_args(known_only=True)

# set args to wandb
if args.use_wandb:
    wandb.init(project="xdocvqa_v2", config=args)

if args.model == "pix2struct":
    model, processor = get_pix2struct_model(task="vqa")
elif args.model == "donut":
    model, processor = get_donut_model()

# get the dataloaders
train_dataloader, val_dataloader, _ = get_loaders(processor, args.data_path, args.batch_size)
epochs = args.num_epochs

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

def train_one_epoch(epoch, data_loader, save_path):
    model.train()
    print("Epoch:", epoch)
    running_loss_text_1 =  0
    for idx, batch in enumerate(data_loader):
        
        if args.model == "pix2struct":
            labels = batch.pop("labels").to(device)
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)

            optimizer.zero_grad()
            outputs1 = model(flattened_patches=flattened_patches,
                                            attention_mask=attention_mask,
                                            labels=labels)
        elif args.model == "donut":
            labels = batch.pop("labels_donut").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            decoder_input_ids = batch.pop("decoder_input_ids").to(device)
            optimizer.zero_grad()
            outputs1 = model(pixel_values, decoder_input_ids=decoder_input_ids[:, :-1], labels=labels[:, 1:])
        
        
        loss_text_1 = outputs1.loss
        running_loss_text_1 += loss_text_1.item()
        
        show_every = 5
        if idx % show_every == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{idx}/{len(data_loader)}] - "
                f"Loss Text 1: {running_loss_text_1 / show_every:.4f}")
            
            running_loss_text_1 = 0
            
        if idx % 100 ==0:
            print("Saving model")
            torch.save(model.state_dict(), os.path.join(save_path,"last_model_p2x_original.pth"))
        
        if args.use_wandb:
            wandb.log({"loss_text_1": loss_text_1.item()})
        loss_text_1.backward()
        optimizer.step()
        if idx >=500 or epoch > 0:
            scheduler.step()

def eval_docvqa(model):
    model.eval()
    anls = 0
    acc = 0
    batch_idx = 0
    for batch in tqdm(val_dataloader):
        if args.model == "donut":
             pred_answers = pred_donut_model(model, batch, processor)
        elif args.model == "pix2struct":
            flattened_patches = batch.pop("flattened_patches").to(device)
            attention_mask = batch.pop("attention_mask").to(device)
            with torch.no_grad():
                predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask,  max_length=100)
            pred_answers = processor.batch_decode(predictions, skip_special_tokens=True)
        gt_answers = batch.pop("gt_answers")
        metric = evaluator.get_metrics(gt_answers, pred_answers)

        batch_acc = np.mean(metric['accuracy'])
        batch_anls = np.mean(metric['anls'])
        acc += batch_acc
        anls += batch_anls
        batch_idx += 1

    if args.use_wandb:
        wandb.log({"acc_doqvqa": acc / batch_idx, "anls_doqvqa": anls / batch_idx})
    return acc / batch_idx, anls / batch_idx


def pred_donut_model(model, batch, processor, return_probs=False):
    import re
    model.eval()
    pixel_values = batch['pixel_values']
    decoder_input_ids = batch['decoder_input_ids']
    out_seqs = []
    with torch.no_grad():
        for i in range(len(pixel_values)): # there is a problem in donut with generating a batch!
            curr_pixel_values = pixel_values[i].unsqueeze(0)
            # remove padding from decoder_input_ids, padding value is 1
            curr_decoder_input_ids = decoder_input_ids[i]
            curr_decoder_input_ids = curr_decoder_input_ids[curr_decoder_input_ids != 1]
            curr_decoder_input_ids = curr_decoder_input_ids.unsqueeze(0)
            outputs = model.generate(
                                    curr_pixel_values.to(device),
                                    decoder_input_ids=curr_decoder_input_ids.to(device),
                                    max_length=model.decoder.config.max_position_embeddings,
                                    pad_token_id=processor.tokenizer.pad_token_id,
                                    eos_token_id=processor.tokenizer.eos_token_id,
                                    use_cache=True,
                                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                    return_dict_in_generate=True,
                                )
            out_seqs.append(processor.tokenizer.batch_decode(outputs.sequences)[0])
    
    predictions = []
    
    for seq in out_seqs:
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = seq.split("<s_answer>")[-1].strip()  # remove question part
        seq = re.sub(r"<.*?>", "", seq).strip()
        predictions.append(seq)
    return predictions

if __name__ == "__main__":
    loss_bce = torch.nn.BCELoss() 
    evaluator = Evaluator(case_sensitive=False)
    best_anls = 0
    print("Training for ", epochs, " epochs")
    save_path = os.path.join(args.save_path, args.experiment_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in range(epochs):
        print("Training epoch: ", epoch, " /  Total epochs: ", epochs)
        train_one_epoch(epoch, train_dataloader, save_path)
        print("Evaluation docvqa epoch: ", epoch)
        acc, anls = eval_docvqa(model)
        print("acc: ", acc, "\n anls: ", anls)

        if anls >= best_anls:
            best_anls = anls
            print("Saving model")
            torch.save(model.state_dict(), os.path.join(save_path, "best_model_docvqa_original.pth"))
        print("Saving model")
        torch.save(model.state_dict(), os.path.join(save_path,"last_model_docvqa_original.pth"))
