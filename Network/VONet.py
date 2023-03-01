# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import torch 
import torch.nn as nn
import torch.nn.functional as F
from .PWC import PWCDCNet as FlowNet
from .VOFlowNet import VOFlowRes as FlowPoseNet
from params import par
import numpy as np
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
from model import IMUKalmanFilter
class VONet(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm):
        super(VONet, self).__init__()

        self.flowNet     = FlowNet()
        self.flowPoseNet = FlowPoseNet()

        tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        tmp = self.cnn(tmp)

        # RNN
        if par.hybrid_recurrency and par.enable_ekf:
            lstm_input_size = IMUKalmanFilter.STATE_VECTOR_DIM ** 2 + IMUKalmanFilter.STATE_VECTOR_DIM
        else:
            lstm_input_size = 0
        self.rnn = nn.LSTM(
                input_size=int(np.prod(tmp.size())) + lstm_input_size,
                hidden_size=par.rnn_hidden_size,
                num_layers=par.rnn_num_layers,
                dropout=par.rnn_dropout_between,
                batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=12)

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  # orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  # orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n // 4, n // 2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def stack_img(self, images):
        # images: (batch, seq_len, channel, width, height)
        #stack img to 2 consecutive img
        images = torch.cat((images[:, :-1], images[:, 1:]), dim=2)
        batch_size = images.size(0)
        seq_len = images.size(1)
        images = images.view(batch_size * seq_len, images.size(2), images.size(3), images.size(4))

        x = self.forward(images)
        
        x = x.view(batch_size, seq_len, -1)
        return images

    def forward_one_ts(self, feature_vector, lstm_init_state=None):

        # lstm_init_state has the dimension of (# batch, 2 (hidden/cell), lstm layers, lstm hidden size)
        if lstm_init_state is not None:
            hidden_state = lstm_init_state[:, 0, :, :].permute(1, 0, 2).contiguous()
            cell_state = lstm_init_state[:, 1, :, :].permute(1, 0, 2).contiguous()
            lstm_init_state = (hidden_state, cell_state,)

        # RNN
        # lstm_state is (hidden state, cell state,)
        # each hidden/cell state has the shape (lstm layers, batch size, lstm hidden size)

        out, lstm_state = self.rnn(feature_vector.unsqueeze(1), lstm_init_state)
        out = self.rnn_drop_out(out)
        out = self.linear(out)

        # rearrange the shape back to (# batch, 2 (hidden/cell), lstm layers, lstm hidden size)
        lstm_state = torch.stack(lstm_state, dim=0)
        lstm_state = lstm_state.permute(2, 0, 1, 3)

        return out.squeeze(1), lstm_state

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        flow = self.flowNet(x[0:2])
        flow_input = torch.cat( ( flow, x[2] ), dim=1 )        
        feature = self.flowPoseNet(flow_input)

        return feature
    

