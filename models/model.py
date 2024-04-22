import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
import copy
from dataclasses import field

def construct_batched_edge_index(self, mask, num_nodes):
    B = mask.shape[0]
    edge_index = []

    for b in range(B):
        offset = b * num_nodes
        for i in range(num_nodes):
            if mask[b, i]:
                continue
            for j in range(num_nodes):
                if i != j and not mask[b, j]:
                    edge_index.append([offset + i, offset + j])

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index_tensor.to(mask.device)  # Move edge_index to the same device as the mask

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Copied and modified from PyTorch source to return attention weights
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if self.activation == F.relu:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # activation function is gelu
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src), True)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, self.self_attn.attn_weights  # Return attention weights here

class ModelEMA:
    def __init__(self, model, decay):
        self.ema_model = copy.deepcopy(model)
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            ema_params = self.ema_model.state_dict().items()
            model_params = model.state_dict().items()
            for (ema_name, ema_param), (model_name, model_param) in zip(ema_params, model_params):
                if "num_batches_tracked" in model_name:
                    ema_param = model_param
                else:
                    try:
                        ema_param.mul_(self.decay).add_(model_param, alpha=1 - self.decay)
                    except Exception as e:
                        print(e)
                        print(ema_name, model_name)
                        raise

class ResidualGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6):
        super(ResidualGATLayer, self).__init__()
        self.conv = geom_nn.GATConv(in_channels, out_channels, heads=heads, dropout=dropout)
        self.skip_connection = nn.Identity()

    def forward(self, x, edge_index):
        identity = self.skip_connection(x)
        x = F.elu(self.conv(x, edge_index))
        x += identity  # Add the input (residual connection)
        return x
    
class ResidualGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, dropout=0.6):
        super(ResidualGCNLayer, self).__init__()
        self.conv = geom_nn.GCNConv(in_channels, out_channels, improved=improved)
        self.dropout = nn.Dropout(dropout)

        # Use nn.Identity() for matching dimensions, otherwise use a Linear layer for transformation
        if in_channels != out_channels:
            self.skip_connection = nn.Linear(in_channels, out_channels)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, edge_index):
        identity = self.skip_connection(x)
        x = F.relu(self.conv(x, edge_index))
        x = self.dropout(x)
        x += identity  # Residual connection
        return x
    
class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6):
        super(GATLayer, self).__init__()
        self.conv = geom_nn.GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        return F.elu(self.conv(x, edge_index))

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv = geom_nn.GCNConv(in_channels, out_channels, improved=True)

    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))

def load_weights(model, pth_file, except_param=None):
    # 加载.pth文件中的参数
    state_dict = torch.load(pth_file, map_location=torch.device('cpu'))['model_ema']
    
    # 获取模型的参数字典
    model_dict = model.state_dict()

    # 更新模型参数
    for name, param in state_dict.items():
        if name in model_dict:
            if except_param and except_param in name:
                print(f"param '{name}' not loaded.")
            elif param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print(f"param '{name}' not loaded.", param.size(), model_dict[name].size())

    # 检查是否有未加载的参数
    missed_params = [name for name in model_dict if name not in state_dict]
    if missed_params:
        print("missing params:")
        for name in missed_params:
            print(name)

class RegressoionHead(nn.Module):
    
    def __init__(self, d_embedding):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding//2)
        self.layer2 = nn.Linear(d_embedding//2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

class HybridModelV4(nn.Module):
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
                 dropout_rate=0.0,
                 c_hidden_size=32):
        super(HybridModelV4, self).__init__()
        # Decide the model type based on the argument
        self.use_cls_token = cls_token
        self.use_mlp = model_type == "mlp"
        self.use_transformer = model_type == "transformer"
        self.use_gcn = model_type == "gcn" or model_type == "gat" or model_type == "rgcn" or model_type == "rgat" 
        self.use_lstm = model_type == "lstm"
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
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_head,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout_rate
                ),
                num_layers=num_layers
            )
        elif self.use_lstm:
            # Define LSTM
            self.embedding = nn.Linear(input_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        else:
            raise NotImplementedError
        
        self.dropout = nn.Dropout(dropout_rate)
        # Layer for regression
        self.fc_condition = nn.Linear(num_conditions, c_hidden_size)
        
        self.fc_regression = RegressoionHead(hidden_size)
        
        self.fc_classification = nn.Linear(hidden_size, num_classes)
        self.fc_classification2 = nn.Linear(hidden_size, 3)
        self.fc_concat = nn.Linear(hidden_size + c_hidden_size, hidden_size)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(hidden_size)
            
    def forward(self, x, c_inputs, mask=None, return_cls=False):
        B, N, D = x.shape
        if self.use_transformer:
            # Transformer forward pass
            x = self.embedding(x)
            if self.use_cls_token:
                assert not self.average_feats
                cls_token = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                mask = torch.cat([torch.zeros((B, 1), dtype=torch.bool).to(mask.device), mask], dim=1)
            x = x.permute(1, 0, 2)  # (S, N, E)
            transformer_out = self.transformer(x, src_key_padding_mask=mask)
            if self.average_feats:
                unmask_weight = torch.logical_not(mask).float().transpose(0, 1)
                x = transformer_out * unmask_weight.unsqueeze(-1)
                x = x.sum(0) / unmask_weight.sum(0).unsqueeze(-1)
            else:
                x = self.to_cls_token(transformer_out[0])
        elif self.use_lstm:
            x = self.embedding(x)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            x = out[:, -1, :]
        else:
            raise NotImplementedError

        if return_cls:
            return x
        
        # Concatenate with c_inputs 
        x = self.fc_concat(torch.cat([self.fc_condition(c_inputs), x], -1))
        if self.with_bn:
            x = self.bn(x)
        x = torch.relu(x)

        # Classification outputs
        out = self.fc_regression(x)
        out2 = self.fc_classification2(x)
        out_cls = self.fc_classification(x)

        return out, out2, out_cls
    
class HybridModelV5(nn.Module):
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
                 dropout_rate=0.0,
                 c_hidden_size=32):
        super(HybridModelV5, self).__init__()
        # Decide the model type based on the argument
        self.use_cls_token = cls_token
        self.use_mlp = model_type == "mlp"
        self.use_transformer = model_type == "transformer"
        self.use_gcn = model_type == "gcn" or model_type == "gat" or model_type == "rgcn" or model_type == "rgat" 
        self.use_lstm = model_type == "lstm"
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
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_head,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout_rate
                ),
                num_layers=num_layers
            )
        elif self.use_lstm:
            # Define LSTM
            self.embedding = nn.Linear(input_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        else:
            raise NotImplementedError
        
        self.dropout = nn.Dropout(dropout_rate)
        # Layer for regression
        self.fc_condition = nn.Linear(num_conditions, c_hidden_size)
        
        self.fc_regression = RegressoionHead(hidden_size)
        
        self.fc_classification = nn.Linear(hidden_size, num_classes)
        self.fc_classification2 = nn.Linear(hidden_size, num_classes)
        self.fc_concat = nn.Linear(hidden_size + c_hidden_size, hidden_size)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(hidden_size)
            
    def forward(self, x, c_inputs, mask=None, return_cls=False):
        B, N, D = x.shape
        if self.use_transformer:
            # Transformer forward pass
            x = self.embedding(x)
            if self.use_cls_token:
                assert not self.average_feats
                cls_token = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                mask = torch.cat([torch.zeros((B, 1), dtype=torch.bool).to(mask.device), mask], dim=1)
            x = x.permute(1, 0, 2)  # (S, N, E)
            transformer_out = self.transformer(x, src_key_padding_mask=mask)
            if self.average_feats:
                unmask_weight = torch.logical_not(mask).float().transpose(0, 1)
                x = transformer_out * unmask_weight.unsqueeze(-1)
                x = x.sum(0) / unmask_weight.sum(0).unsqueeze(-1)
            else:
                x = self.to_cls_token(transformer_out[0])
        elif self.use_lstm:
            x = self.embedding(x)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            x = out[:, -1, :]
        else:
            raise NotImplementedError

        if return_cls:
            return x
        
        # Concatenate with c_inputs 
        x = self.fc_concat(torch.cat([self.fc_condition(c_inputs), x], -1))
        if self.with_bn:
            x = self.bn(x)
        x = torch.relu(x)

        # Classification outputs
        out = self.fc_regression(x)
        out2 = self.fc_classification2(x)
        out_cls = self.fc_classification(x)

        return out, out2, out_cls