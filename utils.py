import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
from einops import rearrange
import torchvision.transforms.functional as TF


def find_bounding_boxes(matrix):
    
    num_labels, labels = cv2.connectedComponents(matrix, connectivity=4)

    bounding_boxes = []

    for label in range(1, num_labels):
        coords = np.column_stack(np.where(labels == label))
        
        component_mask = (labels == label)
        
        unique_values = np.unique(matrix[component_mask])
        
        for value in unique_values:
            value_mask = component_mask & (matrix == value)
            
            if np.any(value_mask):
                value_coords = np.column_stack(np.where(value_mask))
                
                top_left = value_coords.min(axis=0)
                bottom_right = value_coords.max(axis=0) + 1
                
                bounding_boxes.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

    return bounding_boxes


def max_n(tensor, n):
    max_values = []
    for i in range(n):
        max_val = torch.max(tensor)
        max_values.append(max_val.item())
        tensor = tensor[tensor != max_val]
    return max_values

def post_process_mask_2(mask, mask_2, p1, p2, k):
    mask[mask_2] = 0.0
    patches = np.array(p1)*np.array(p2)
    
    for i in range(len(patches)):
        t_img_m = rearrange(mask[i,:patches[i],:].unsqueeze(0), 'b (h w) (p1 p2) -> b (h p1) (w p2)',
                                        p1 = 1, p2 = 1,  h=p1[i]).cpu()
        bounding_boxes = find_bounding_boxes(t_img_m.squeeze().detach().numpy()[:,:].astype(np.uint8))
    
        new_img = torch.zeros_like(t_img_m)
        maxes = []

        for _, box in enumerate(bounding_boxes):
            max_box = t_img_m[:, box[0]:box[2], box[1]:box[3]].max().item()
            maxes.append(max_box)
        
        # keep only to k boxes
        maxes = np.array(maxes)
        maxes = np.sort(maxes)
        THRESH = maxes[-k]
        for _, box in enumerate(bounding_boxes):
            if t_img_m[:, box[0]:box[2], box[1]:box[3]].max().item() >= THRESH:
                new_img[:, box[0]:box[2], box[1]:box[3]] = 1
            else:
                new_img[:, box[0]:box[2], box[1]:box[3]] = 0
        

        t_img_m_back = rearrange(new_img, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = 1, p2 = 1,  h=p1[i])
        mask[i,:patches[i],:] = t_img_m_back[0].clone()
        mask[i,patches[i]:,:]  = 0.
        mask[i,mask_2[i]] = 1.0
    
    return mask


def post_process_mask(mask, attn_0, p1, p2, k, img_header=None):
    
    if img_header is not None:
        mask[img_header] = 0.0
    patches = np.array(p1)*np.array(p2)
    
    for i in range(len(patches)):
        t_img_m = rearrange(mask[i,:patches[i],:].unsqueeze(0), 'b (h w) (p1 p2) -> b (h p1) (w p2)',
                                        p1 = 1, p2 = 1,  h=p1[i]).cpu()
        bounding_boxes = find_bounding_boxes(t_img_m.squeeze().detach().numpy()[:,:].astype(np.uint8))
    
        THRESH = max_n(attn_0[i], k)[-1]
        att_img_thresh = (attn_0[i]>=THRESH).float()
        # transform every 16 by 16 pixels of att_img into 1 average pixel
        att_img_small = att_img_thresh.reshape(3, p1[i], p2[i], 16, 16).mean(-1).mean(-1).mean(0).unsqueeze(0)
        att_img_thresh = (att_img_small).float()
    
        new_img = torch.zeros_like(t_img_m)

        for _, box in enumerate(bounding_boxes):
            if att_img_thresh[:, box[0]:box[2], box[1]:box[3]].sum() > 0:
                new_img[:, box[0]:box[2], box[1]:box[3]] = 1
        t_img_m_back = rearrange(new_img, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = 1, p2 = 1,  h=p1[i])
        mask[i,:patches[i],:] = t_img_m_back[0].clone()
        mask[i,patches[i]:,:]  = 0.
        if img_header is not None:
            mask[i,img_header[i]] = 1.0
    
    return mask

def patches_to_img(flattened_patches, plotted_flattened_patches, mean=[0.9087862675462285, 0.9087862675462285, 0.9087862675462285], std=[0.28134391081140947, 0.28134391081140947, 0.28134391081140947]):
    
    p1 = int(max(list(flattened_patches[:,:,0].cpu().numpy())[0]))
    p2 = int(max(list(flattened_patches[:,:,1].cpu().numpy())[0]))
    patches = p1*p2
    t_img = rearrange(plotted_flattened_patches[:,:patches,2:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                p1 = 16, p2 = 16,  h=p1).cpu()
    t_img = t_img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)

    t_img = t_img.squeeze(0)
    pil_image = TF.to_pil_image(t_img)
    
    return pil_image

def find_bg(features, mask_out):
    features = torch.round(features[:,:, 2:] * 10000) / 10000
    for i in range(features.size(1)):
        if torch.var(features[:,i]) <= 0.01:
            mask_out[:,i] = 0.0
    return mask_out

def find_bg_2(np_img):
    window_size = (32, 32)
    bg_mask = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=np.uint8)

    for i in range(0, np_img.shape[0], window_size[0]):
        for j in range(0, np_img.shape[1], window_size[1]):
            window = np_img[i:i + window_size[0], j:j + window_size[1]]
            if np.var(window) < 0.01: # threshold for background detection
                bg_mask[i:i + window_size[0], j:j + window_size[1]] = 1

    return bg_mask

def build_model(model_name, task='vqa', args=None):
    if model_name == 'pix2struct':
        from models.pix2struct import get_pix2struct_model
        from models.model import XDocVQA
        pix2struct, processor = get_pix2struct_model(task=task, ckpts=args.base_pix2struct_path)
        model = XDocVQA(pix2struct_model=pix2struct, max_patches=args.max_patches)
    elif model_name == 'donut':
        from models.donut import get_donut_model
        from models.model import XDocVQADonut
        donut, processor = get_donut_model(ckpts="/data/....../original.pth")
        model = XDocVQADonut(donut_model=donut)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model, processor

class CustomLoss:
    def __init__(self, model_type=None):
        self.model_type = model_type

    def continuity_loss_2d(self, mask, pos_encoding):
        mask = mask.squeeze()
        B, N= mask.shape

        pos_encoding = pos_encoding.to(torch.long) - 1

        rows = pos_encoding[..., 0].max(dim=1).values + 1
        cols = pos_encoding[..., 1].max(dim=1).values + 1
        max_rows = rows.max().item()
        max_cols = cols.max().item()

        matrix = torch.zeros((B, max_rows, max_cols), dtype=mask.dtype, device=mask.device)

        batch_indices = torch.arange(B, device=mask.device).unsqueeze(1).expand(-1, N)
        matrix[batch_indices, pos_encoding[..., 0], pos_encoding[..., 1]] = mask

        continuity_x = torch.linalg.norm(matrix[:, :, :-1] - matrix[:, :, 1:], ord=1, dim=(1, 2))
        continuity_y = torch.linalg.norm(matrix[:, :-1, :] - matrix[:, 1:, :], ord=1, dim=(1, 2))

        total_pixels = (rows * cols).float()
        loss = (continuity_x + continuity_y) / (total_pixels - 1)
        
        return loss.mean()

    def mask_loss(self, predicted_mask, pos_encoding=None):
        flattened = predicted_mask.flatten(start_dim = 1)

        if self.model_type == "donut":
            # Flatten the mask to 1D
            l1_loss = torch.mean(torch.abs(predicted_mask))
            return l1_loss, l1_loss*0, l1_loss*0
        else:
            # L1 Regularization to encourage minimality and sparsity
            l1_reg = 10 * torch.linalg.norm(flattened.transpose(0,1), ord=1)/flattened.numel() # give more importance to sparsity
            
            # 2D Continuity Regularization to encourage smoothness
            continuity_reg = self.continuity_loss_2d(flattened, pos_encoding)
            
            total_loss = l1_reg + continuity_reg

            return total_loss, l1_reg, continuity_reg

    def KD_loss(self, outputs1, outputs2):
        logits_1 = outputs1.logits
        logits_2 = outputs2.logits
        # use KL divergence loss
        distillation_loss = F.kl_div(F.log_softmax(logits_1, dim=-1), F.softmax(logits_2, dim=-1))
        return distillation_loss

class STE_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(threshold)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        # the 'straight-through' part - just pass the gradient back
        # (but in practice we clamp it to the [-1,1] range for stability)
        threshold = ctx.saved_tensors[0]
        return F.hardtanh(grad_output - threshold), None # threshold itself does not have a gradient

# another version, "sigmoid-adjusted straight-through", that uses the sigmoid derivative in the backward pass:
class SAST_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, threshold)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        sigmoid_grad = torch.sigmoid(input - threshold) * (1 - torch.sigmoid(input - threshold))
        return grad_output * sigmoid_grad, None # threshold does not have a gradient


class ThresholdStraightThrough(nn.Module):
    def __init__(self, threshold=0., grad = 'identity'):
        # grad should be one of 'identity' or 'sigmoid'
        super(ThresholdStraightThrough, self).__init__()
        self.threshold = torch.autograd.Variable(torch.Tensor([threshold])) # for the step function
        
        if grad == 'identity':
            self.use_sigmoid = False
        elif grad == 'sigmoid':
            self.use_sigmoid = True
        else:
            raise ValueError("ThresholdStraighThrough requires 'grad' arg to be one of: 'identity', 'sigmoid'")

    def forward(self, x):
        self.threshold = self.threshold.to(x.device)
        if self.use_sigmoid:
            return SAST_Func.apply(x, self.threshold)
        else:
            return STE_Func.apply(x, self.threshold)


if __name__ == '__main__':
    
    # test straight-through estimator:
    print('SAST')
    x = torch.arange(-5, 5, 1.0, requires_grad=True)
    ste = ThresholdStraightThrough(grad='sigmoid', threshold=0.5)
    y = ste(x)

    y.backward(torch.ones_like(x))
    for x_val, y_val, grad_val in zip(x, y, x.grad):
        print(f'x:{x_val:<5}  y:{y_val:<5}  grad:{grad_val:<5.5f}')

    # compare sigmoid:
    print(f'sigmoid')
    x = torch.arange(-5, 5, 1.0, requires_grad=True)
    ste = ThresholdStraightThrough(grad='sigmoid', threshold=0.5)
    y = torch.sigmoid(ste(x))

    y.backward(torch.ones_like(x))
    for x_val, y_val, grad_val in zip(x, y, x.grad):
        print(f'x:{x_val:<5}  y:{y_val:<5.2f}  grad:{grad_val:<5.5f}')
