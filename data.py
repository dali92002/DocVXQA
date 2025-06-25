from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image, ImageChops
import os 
import torch
import random
from transformers import AutoProcessor, Pix2StructProcessor
from arg_utils import Args
import numpy as np
import torchvision.transforms as transforms

args = Args().parse_args(known_only=True)

processor = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")
processor_renderer = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
    
class SPDocVQADataset(Dataset):
    def __init__(self, data_path, set, processor, project_mode=None, use_proc_data = args.use_proc_data) -> None:
        super().__init__()
        self.data_path = data_path
        if args.dataset == "infographics_vqa":
            self.data_json = json.load(open(os.path.join(data_path, 'infographicsvqa_qas', set), "r"))
            self.dataset = self.data_json["data"]
        elif args.dataset == "SPDocVQA":
            self.data_json = json.load(open(os.path.join(data_path, 'qas', set), "r"))
            self.dataset = self.data_json["data"]
        elif args.dataset == "pfl_docvqa":
            imdb_npy_path = os.path.join(data_path, f"{set}.npy")
            # read .npy imdb file
            self.data_imdb = np.load(imdb_npy_path, allow_pickle=True)
            self.dataset = self.data_imdb[1:]
        if args.dataset == "pfl_docvqa":
            self.pixlel_ratios = json.load(open('/data/users/msouibgui/datasets/pfl_docvqa/ratios/ratios_all.json', "r"))
        else:
            self.pixlel_ratios = json.load(open(os.path.join(data_path,'ratios','ratios_all.json'), "r"))
        self.processor = processor
        self.project_mode = project_mode
        self.use_proc_data = use_proc_data
        self.colpali_mode = args.colpali_mode
        self.img_transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
            ])
        self.set = set
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        if self.project_mode == "alpha":
            index = 33
        data_point = self.dataset[index]
        data_dict = {}
        if self.use_proc_data:
            processed_data = torch.load(os.path.join(self.data_path, "processed", str(data_point["questionId"])+".pt"))
            data_dict["flattened_patches"] = processed_data["flattened_patches"]
            data_dict["attention_mask"] = processed_data["attention_mask"]
        
            img = Image.open(os.path.join(self.data_path, data_point["image"]))
            question = data_point["question"]
            
            data_dict["question"] = question
            
        else:
            if args.dataset == "infographics_vqa":
                img = Image.open(os.path.join(self.data_path, "infographicsvqa_images", data_point["image_local_name"])) #.convert("RGB")                
                data_dict["answer_type"] = data_point["answer_type"]
            elif args.dataset == "SPDocVQA":
                img = Image.open(os.path.join(self.data_path, data_point["image"]))
            elif args.dataset == "pfl_docvqa":
                img_folder = "full" if self.set == "imdb_test" else "blue"
                img = Image.open(os.path.join(f"/data-net/125/shared/PFL-DocVQA/images/{img_folder}/", data_point["image_name"])+".jpg") #.convert("RGB")
                data_point["questionId"] = str(index) + "__"+data_point['image_name']
            
            question = data_point["question"]
            data_dict["img"] = img
            data_dict["question"] = question   
            
        
        if args.model == "donut":
            self.max_length = 64
            question = data_point["question"]
            
            pixel_values = self.processor(img.convert("RGB"), return_tensors="pt").pixel_values
            input_tensor = pixel_values.squeeze()

            # input_ids
            answers = list(set(answer.lower() for answer in data_point["answers"]))
            answer = random.choice(answers)

            # prepare prompt
            if "train" in self.set:
                task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>{gt_answer}</s_answer></s>"
                prompt = task_prompt.replace("{user_input}", question)
                prompt = prompt.replace("{gt_answer}", answer)
            else:
                task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
                prompt = task_prompt.replace("{user_input}", question)
            input_ids = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True).input_ids.squeeze(0)


            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels = labels.unsqueeze(0)

            for _i in range(len(labels)):
                labels[_i][:torch.nonzero(labels[_i] == processor.tokenizer.convert_tokens_to_ids('<s_answer>')).sum() + 1] = -100

            labels = labels.squeeze()
            data_dict["img"] = input_tensor
            data_dict["question"] = question
            data_dict["input_ids"] = input_ids
            data_dict["labels"] = labels

        if self.pixlel_ratios is not None:
            mask_question = Image.open(os.path.join(args.colpali_path, str(data_point["questionId"]) + ".png")).convert("L")
            mask_answer = Image.open(os.path.join(args.colpali_path, str(data_point["questionId"]) + "_answer.png")).convert("L")

            overal_mask = ImageChops.add(mask_question, mask_answer)
            overal_mask = overal_mask.point(lambda p: min(p, 255))  

            ratio = self.pixlel_ratios[str(data_point["questionId"])][0]
            new_height = int(ratio * overal_mask.height) + overal_mask.height
            mask_headed = Image.new("L", (overal_mask.width, new_height), 255)  # make the top part white
            mask_headed.paste(overal_mask, (0, int(ratio * overal_mask.height)))
            data_dict["mask_img"] = self.img_transform(mask_headed)
            data_dict["header_ratio"] = ratio

        if args.dataset == "pfl_docvqa":
            data_dict["gt_answers"] = [data_point["answers"].lower()]
            data_dict["text"] = data_point["answers"].lower()
        else:
            answers = list(set(answer.lower() for answer in data_point["answers"]))
            text = random.choice(answers)
            data_dict["gt_answers"] = answers
            data_dict["text"] = text
        data_dict["questionId"] = data_point["questionId"]
        return data_dict      

def collator(batch):
    if args.use_proc_data:
        encodings = {}
        flattened_patches = [item["flattened_patches"].squeeze() for item in batch]
        attention_mask = [item["attention_mask"].squeeze() for item in batch]
        
        encodings["flattened_patches"] =  torch.stack(flattened_patches)
        encodings["attention_mask"]  = torch.stack(attention_mask)
    else:
        if args.model == "pix2struct":
            imgs = [item["img"] for item in batch]
            questions = [item["question"] for item in batch]
            encodings = processor_renderer(images=imgs, text=questions, return_tensors="pt", add_special_tokens=True, max_patches=args.max_patches, padding=True)
        elif args.model == "donut":
            encodings = {}
            input_imgs = [item["img"] for item in batch]
            encodings["pixel_values"] = torch.stack(input_imgs)
            decoder_input_ids = [item["input_ids"] for item in batch]
            encodings["decoder_input_ids"] = torch.stack(decoder_input_ids)
            labels_donut = [item["labels"] for item in batch]
            encodings["labels_donut"] = torch.stack(labels_donut)
            
    question_ids = [item["questionId"] for item in batch]     
    encodings["question_ids"] = question_ids

    target_mask_imgs = [item["mask_img"] for item in batch]
    encodings["target_mask_imgs"] =  torch.stack(target_mask_imgs)

    if "header_ratio" in batch[0]:
        header_ratios = [item["header_ratio"] for item in batch]
        encodings["header_ratios"] = torch.tensor(header_ratios)
    if "answer_type" in batch[0]:
        answer_types = [item["answer_type"] for item in batch]
        encodings["answer_types"] = answer_types

    texts = [item["text"] for item in batch]
    text_inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True, max_length=args.max_length)
    encodings["labels"] = text_inputs.input_ids
    gt_answers = [item["gt_answers"] for item in batch]
    encodings["gt_answers"] = gt_answers
    
    return encodings


def get_loaders(processor, data_path, batch_size=1):
    if args.dataset == "infographics_vqa":
        train_dataset = SPDocVQADataset(data_path, "infographicsVQA_train_v1.0.json", processor)
        val_dataset = SPDocVQADataset(data_path, "infographicsVQA_val_v1.0_withQT.json", processor)
        test_dataset = SPDocVQADataset(data_path, "infographicsVQA_test_v1.0.json", processor)
    elif args.dataset == "SPDocVQA":
        train_dataset = SPDocVQADataset(data_path, "train_v1.0_withQT.json", processor)
        val_dataset = SPDocVQADataset(data_path, "val_v1.0_withQT.json", processor)
        test_dataset = SPDocVQADataset(data_path, "test_v1.0_withQT.json", processor)
    elif args.dataset == "pfl_docvqa":
        train_dataset = SPDocVQADataset(data_path, "imdb_train", processor)
        val_dataset = SPDocVQADataset(data_path, "imdb_val", processor)
        test_dataset = SPDocVQADataset(data_path, "imdb_test", processor)
    if args.debug_mode:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(2000))
        val_dataset = Subset(val_dataset, range(500))
        test_dataset = Subset(test_dataset, range(500))
    if args.project_mode == "alpha":
        from torch.utils.data import Subset
        val_dataset = SPDocVQADataset(data_path, "val_v1.0_withQT.json", processor, project_mode="alpha")
        train_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, collate_fn=collator)
        sub_val_dataset = Subset(val_dataset, range(1))
        val_dataloader = DataLoader(sub_val_dataset, shuffle=False, batch_size=batch_size*12, collate_fn=collator)

        return train_dataloader, val_dataloader
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator, num_workers=2)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size*4, collate_fn=collator, num_workers=2)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size*4, collate_fn=collator, num_workers=2)
    
    return train_dataloader, val_dataloader, test_dataloader

if __name__=="__main__":

    # dummy test ...
    from transformers import Pix2StructProcessor
    processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
    data_tr = SPDocVQADataset(args.data_path, "train_v1.0_withQT.json", processor)
    data_te = SPDocVQADataset(args.data_path, "test_v1.0_withQT.json", processor)
    data_va = SPDocVQADataset(args.data_path, "val_v1.0_withQT.json", processor)
    
    print(data_te[800])
    print(data_tr[800])
    print(data_va[800])

