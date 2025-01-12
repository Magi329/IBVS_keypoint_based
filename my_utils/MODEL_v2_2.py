import torch
import torch.nn as nn


class QuaternionAlignmentTransformer(nn.Module):
    def __init__(self, hidden_dim, hidden_depth_dim, num_heads, num_layers, input_dim=7, output_dim=4):
        super(QuaternionAlignmentTransformer, self).__init__()

        # Define input embeddings
        self.previous_y_embed = nn.Linear(input_dim, hidden_dim)  # previous_y: 7D (x, y, z, quaternion_w, quaternion_x, quaternion_y, quaternion_z)
        self.x_embed = nn.Linear(28, hidden_dim)  # x: points coordinates in pixel space
        self.delta_trans_embed = nn.Linear(3, hidden_dim)  # delta_trans: (delta_x, delta_y, delta_z)
        self.depth_embed = nn.Linear(1, hidden_depth_dim)  # depth: scalar
        #self.depth0_embed = nn.Linear(1, hidden_depth_dim)  # depth0: scalar

        concate_hidden_dim = hidden_dim*3 + hidden_depth_dim
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=concate_hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # Final layer to output quaternion
        self.fc_out = nn.Linear(concate_hidden_dim, output_dim)  # Output: 7D quaternion

        self.dropout = nn.Dropout(p=0.2)


    def forward(self, previous_y, x, delta_trans, depth):
        # Embed each input feature
        previous_y_emb = self.previous_y_embed(previous_y)
        x_emb = self.x_embed(x)
        delta_trans_emb = self.delta_trans_embed(delta_trans)
        depth_emb = self.depth_embed(depth)
        #depth0_emb = self.depth0_embed(depth0)

        # Concatenate all embeddings along the feature dimension
        embedded_inputs = torch.cat((previous_y_emb, x_emb, delta_trans_emb, depth_emb), dim=-1)

        # Pass through Transformer Encoder
        # We need to reshape the inputs to (seq_len, batch_size, feature_dim)
        embedded_inputs = embedded_inputs.unsqueeze(0)  # Adding a fake batch dimension for sequence length
        embedded_inputs = self.dropout(embedded_inputs) # Add dropout

        transformer_out = self.transformer_encoder(embedded_inputs)

        # Take the output from the last sequence step (i.e., final embedding)
        transformer_out = transformer_out.squeeze(0)

        # Pass through the final linear layer to predict the quaternion
        output_quaternion = self.fc_out(transformer_out)

        return output_quaternion

class Alpha(nn.Module):
    def __init__(self):
        super(Alpha, self).__init__()
        self.fc1 = nn.Linear(28, 128)  # 扩大第一层的宽度
        self.ln1 = nn.LayerNorm(128)  # LN after fc1
        self.fc2 = nn.Linear(128, 256)  # 增加宽度
        self.ln2 = nn.LayerNorm(256)  # LN after fc2
        self.fc3 = nn.Linear(256, 256)  # 添加额外的全连接层
        self.ln3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, 128)  # 缩减宽度，增加网络的层次感
        self.ln4 = nn.LayerNorm(128)
        self.fc5 = nn.Linear(128, 64)
        self.ln5 = nn.LayerNorm(64)
        self.fc6 = nn.Linear(64, 7)  # 输出层，保持不变
        self.dropout = nn.Dropout(p=0.3)  # 提高 Dropout 的丢弃率

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln4(self.fc4(x)))
        x = torch.relu(self.ln5(self.fc5(x)))
        x = self.fc6(x)  
        return x