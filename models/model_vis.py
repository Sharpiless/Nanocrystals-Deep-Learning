import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class RegressoionHead(nn.Module):
    
    def __init__(self, d_embedding):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding//2)
        self.layer2 = nn.Linear(d_embedding//2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

class HybridModelV3(nn.Module):
    def __init__(self, 
                 model_type, 
                 cls_token,
                 num_classes,
                 input_size,
                 hidden_size, 
                 dim_feedforward, 
                 num_layers,
                 num_head,
                 num_conditions,
                 average_feats=False,
                 with_bn=False,
                 dropout_rate=0.2,
                 c_hidden_size=32):
        super(HybridModelV3, self).__init__()
        # Decide the model type based on the argument
        self.use_cls_token = cls_token
        self.use_transformer = model_type == "transformer"
        self.hidden_size = hidden_size
        self.average_feats = average_feats
        self.num_layers = num_layers
        self.with_bn = with_bn
        if self.use_transformer:
            # Define Transformer components
            self.embedding = nn.Linear(input_size, hidden_size)
            if self.use_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.to_cls_token = nn.Identity()
            encoder_layers = CustomTransformerEncoderLayer(
                hidden_size, num_head, dim_feedforward
            )
            self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        else:
            raise NotImplementedError
        
        self.dropout = nn.Dropout(dropout_rate)
        # Layer for regression
        self.fc_condition = nn.Linear(num_conditions, c_hidden_size)
        
        self.fc_regression = RegressoionHead(hidden_size)
        self.fc_regression2 = RegressoionHead(hidden_size)
        
        self.fc_classification = nn.Linear(hidden_size, num_classes)
        self.fc_concat = nn.Linear(hidden_size + c_hidden_size, hidden_size)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(input_size)
    
    def forward(self, x, c_inputs, mask=None, return_attn=False):
        B, N, D = x.shape
        if self.with_bn:
            x = self.bn(x.reshape(B*N, D)).reshape((B, N, D))
        if self.use_transformer:
            # Transformer forward pass
            x = self.embedding(x)
            if self.use_cls_token:
                assert not self.average_feats
                cls_token = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                mask = torch.cat([torch.zeros((B, 1), dtype=torch.bool).to(mask.device), mask], dim=1)
            x = x.permute(1, 0, 2)  # (S, N, E)
            transformer_out, attn_weights = [], []
            for layer in self.transformer.layers:
                x, attn = layer(x, src_key_padding_mask=mask)
                transformer_out.append(x)
                attn_weights.append(attn)
            transformer_out = transformer_out[-1]
            # transformer_out = self.transformer(x, src_key_padding_mask=mask)
            if self.average_feats:
                unmask_weight = torch.logical_not(mask).float().transpose(0, 1)
                x = transformer_out * unmask_weight.unsqueeze(-1)
                x = x.sum(0) / unmask_weight.sum(0).unsqueeze(-1)
            else:
                x = self.to_cls_token(transformer_out[0])
        else:
            raise NotImplementedError

        # Concatenate with c_inputs 
        x = self.fc_concat(torch.cat([self.fc_condition(c_inputs), x], -1))
        # x = self.dropout(torch.relu(x))
        x = torch.relu(x)

        # Classification outputs
        out = self.fc_regression(x)
        out2 = self.fc_regression2(x)
        out_cls = self.fc_classification(x)
        if return_attn:
            return out, out2, out_cls, attn_weights
        else:
            return out, out2, out_cls
