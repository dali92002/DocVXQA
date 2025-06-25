from torch import nn
import torch 
import torch.nn.functional as F
from utils import ThresholdStraightThrough
from einops import rearrange
from torch.nn.functional import interpolate
 
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 4.0) # Initialize biases to positive number

class XDocVQA(nn.Module):
    def __init__(self, pix2struct_model, input_dim=768, vocab_size=50244, seq_len=40, max_patches=1024):
        super(XDocVQA, self).__init__()

        self.pix2struct_model = pix2struct_model
        self.ste = ThresholdStraightThrough(grad='sigmoid', threshold=0.5)
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.max_patches = max_patches
        
        self.mask_head_deocder = nn.Sequential(
                                        nn.Linear(768 * 2 + 2, 1024),
                                        nn.BatchNorm1d(2048),  
                                        nn.LeakyReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.BatchNorm1d(2048),
                                        nn.LeakyReLU(),

                                        nn.Linear(1024, 256),
                                        nn.BatchNorm1d(2048),
                                        nn.LeakyReLU(),  

                                        nn.Linear(256, 1)
                                    )
                                      
    def forward(self, flattened_patches, attention_mask, labels, header_ratios, used_features):
        
        pos_encoding = flattened_patches[:, :, :2]
        p1 = int(max(list(flattened_patches[:,:,0])[0]))
        p2 = int(max(list(flattened_patches[:,:,1])[0]))
        patches = p1*p2
        

        # first pass
        x_out_1 = self.pix2struct_model(flattened_patches=flattened_patches,
                        attention_mask=attention_mask,
                        labels=labels)
        
        # second pass
        if used_features == "enc_dec_attn":
            encoder_output = self.pix2struct_model.encoder(flattened_patches, output_hidden_states=True)
            features = encoder_output['last_hidden_state']
            decoder_input_ids = self.pix2struct_model._shift_right(labels)
            decoder_attention_mask = decoder_input_ids.ne(self.pix2struct_model.config.pad_token_id).float()
            decoder_attention_mask[:, 0] = 1

            decoder_output = self.pix2struct_model.decoder(
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
            all_heads_attn_11 = torch.stack([torch.sum(torch.sum(cross_attention_11[k,:,:i,:self.max_patches], 1), 0).unsqueeze(-1) for k, i in enumerate(lenghts)])
            
            attn_rep = all_heads_attn_11.repeat(1, 1, 768)
            
            feat_attn = torch.cat((pos_encoding, features, attn_rep), 2)
            mask_out = self.mask_head_deocder(feat_attn)
        
        mask_out = F.sigmoid(mask_out) 

        mask_out_img = rearrange(mask_out[:,:patches,:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                p1 = 1, p2 = 1,  h=p1)

        mask_out_img = interpolate(mask_out_img, size=(512, 512), mode='bilinear', align_corners=False)

        x_in_2 = mask_out * flattened_patches
        x_out_2 = self.pix2struct_model(flattened_patches=x_in_2,
                        attention_mask=attention_mask,
                        labels=labels)
        
        return mask_out, x_out_1, x_out_2, all_heads_attn_11, mask_out_img


class XDocVQADonut(nn.Module):
    pass # only local