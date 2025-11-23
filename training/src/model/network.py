import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolution
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
                                 
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # Crop output to match residual length due to padding
        return self.relu2(out[:, :, :res.size(2)] + res)

class DilatedTCN(nn.Module):
    def __init__(self, num_inputs, num_classes, num_channels, kernel_size=2, dropout=0.2):
        super(DilatedTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Causal padding = (kernel - 1) * dilation
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        
        # Adaptive pool to handle variable lengths if needed, but here we just take the last
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x shape: (N, C_in, L)
        y1 = self.tcn(x) # (N, C_out, L)
        
        # Project each time step? Or just the last?
        # Standard TCN usually classifies based on the entire history up to t.
        # But we only need the prediction for the last step t.
        # However, the previous code structure did a Linear projection on flattened output.
        # Let's align with the previous code's head logic:
        
        # Previous code: Project -> Head(AvgPool -> Flatten -> Linear)
        y2 = self.dropout(y1)
        y3 = self.pool(y2).squeeze(-1) # (N, C_out)
        return self.linear(y3)