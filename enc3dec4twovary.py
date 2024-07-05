# enc 3 dec 4 two-stage interpretable with varying knee (noisy feedback)
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import date
from json import dumps
import itertools

torch.set_default_dtype(torch.float32)

identity = str(np.random.random())[2:8]
identity = "enc3dec4two"
print('[ID]', identity)

def get_args(jupyter_notebook):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-code_rate', type=int, default=3)
    parser.add_argument('-block_len', type=int, default=50, help='This do not including zero-padding')
    parser.add_argument('-num_samples_train', type=int, default=80000)
    parser.add_argument('-num_samples_validation', type=int, default=200000)

    parser.add_argument('-feedback_SNR', type=int, default=-3, help='100 means noiseless feeback')
    parser.add_argument('-forward_SNR', type=int, default=0)

    parser.add_argument('-batch_size', type=int, default=400)

    parser.add_argument('-batch_norm', type=bool, default=True, help='True: use batch norm; False: use precalculate norm')

    parser.add_argument('-with_cuda', type=bool, default=False)

    parser.add_argument('-learning_rate', type=float, default=0.02)

    parser.add_argument('-num_epoch', type=int, default=200)

    parser.add_argument('-initial_weights', type=str, default='default')

    if jupyter_notebook:
        args = parser.parse_args(args=[])   # for jupyter notebook
    else:
        args = parser.parse_args()    # in general

    return args

def snr_db_2_sigma(snr_db):
    return 10**(-snr_db*1.0/20)

def errors_ber(y_true, y_pred, device, positions = 'default'):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    y_true = y_true.view(y_true.shape[0], -1, 1)    # the size -1 is inferred from other dimensions
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)
    t1 = torch.round(y_true[:,:,:])
    t2 = torch.round(y_pred[:,:,:])

    # myOtherTensor = np.not_equal(t1, t2).float()
    # k = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    # return k

    comparisin_result = torch.ne(t1, t2).float()  # how many different bits
    if positions == 'default':
        res = torch.sum(comparisin_result)/(comparisin_result.shape[0]*comparisin_result.shape[1])  
    else:   
        res = torch.mean(comparisin_result, dim=0).type(torch.FloatTensor)
        for pos in positions:
            res[pos] = 0.0
        res = torch.mean(res)
    return res

def errors_bler(y_true, y_pred, device, positions = 'default'):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    y_true = y_true.view(y_true.shape[0], -1, 1)    # the size -1 is inferred from other dimensions
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)
    t1 = torch.round(y_true[:,:,:])
    t2 = torch.round(y_pred[:,:,:])

    decoded_bits = t1
    X_test       = t2
    tp0 = (abs(decoded_bits-X_test)).reshape([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    return bler_err_rate

def validation(model, device, X_validation, forward_noise_validation, feedback_noise_validation):
    model.eval()

    rnn_output, codewords, decoder_output = model(X_validation, forward_noise_validation, feedback_noise_validation)
    # ignore the zero padding
    decoder_output = decoder_output[:,:-1,:]
    X_validation = X_validation[:,:-1,:]

    decoder_output = torch.clamp(decoder_output, 0.0, 1.0)
    loss_validation = torch.nn.functional.binary_cross_entropy(decoder_output, X_validation)

    decoder_output = decoder_output.cpu().detach()
    ber_test = errors_ber(decoder_output,X_validation, device)
    bler_test = errors_bler(decoder_output, X_validation, device)

    return loss_validation.item(), ber_test.item(), bler_test.item(), rnn_output, codewords, decoder_output

def train(args, model, device, optimizer, scheduler):  
    model.train()

    loss_train = 0.0

    num_batch = int(args.num_samples_train/args.batch_size)

    for __ in range(num_batch):

        X_train    = torch.randint(0, 2, (args.batch_size, args.block_len, 1))
        X_train = torch.cat([X_train, torch.zeros(args.batch_size, 1, 1)], dim=1)

        forward_noise_train = snr_db_2_sigma(args.forward_SNR)* torch.randn((args.batch_size, args.block_len+1, args.code_rate))

        if args.feedback_SNR == 100:
            feedback_noise_train   = torch.zeros((args.batch_size, args.block_len+1, args.code_rate)) # perfect feedback
        else:
            feedback_noise_train   = snr_db_2_sigma(args.feedback_SNR) * torch.randn((args.batch_size, args.block_len+1, args.code_rate))

        X_train, forward_noise_train, feedback_noise_train = X_train.to(device), forward_noise_train.to(device), feedback_noise_train.to(device)

        optimizer.zero_grad()

        rnn_output, codewords, decoder_output = model(X_train, forward_noise_train, feedback_noise_train)

        decoder_output = torch.clamp(decoder_output, 0.0, 1.0)
        decoder_output = decoder_output[:,:-1,:]
        X_train = X_train[:,:-1,:]
        
        loss = torch.nn.functional.binary_cross_entropy(decoder_output, X_train)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2) # gradient clip 
        optimizer.step()
        scheduler.step()

        loss_train = loss_train + loss.item()

    loss_train = loss_train / num_batch
    return loss_train, X_train, forward_noise_train, feedback_noise_train, rnn_output, codewords, decoder_output

class AE(torch.nn.Module): 
    def __init__(self, args):
        super(AE, self).__init__()

        self.args             = args

        # Encoder
        self.e = torch.nn.Parameter(torch.rand(5), requires_grad = True)
        self.k = torch.nn.Parameter(torch.rand(4), requires_grad = True)
        self.m = torch.nn.Parameter(torch.rand(5), requires_grad = True)


        # Decoder

        # first-stage
        self.d1 = torch.nn.Parameter(torch.rand(2), requires_grad = True)
        self.d2 = torch.nn.Parameter(torch.rand(2), requires_grad = True)
        self.d3 = torch.nn.Parameter(torch.rand(2), requires_grad = True)

        # second-stage
        self.fw1 = torch.nn.Parameter(torch.rand(6), requires_grad = True)
        self.fw2 = torch.nn.Parameter(torch.rand(6), requires_grad = True)
        self.fw3 = torch.nn.Parameter(torch.rand(6), requires_grad = True)
        self.bc1 = torch.nn.Parameter(torch.rand(6), requires_grad = True)
        self.bc2 = torch.nn.Parameter(torch.rand(6), requires_grad = True)
        self.bc3 = torch.nn.Parameter(torch.rand(6), requires_grad = True)


        self.dif1 = torch.nn.Parameter(torch.rand(3), requires_grad = True)
        self.dif2 = torch.nn.Parameter(torch.rand(3), requires_grad = True)
        self.dif3 = torch.nn.Parameter(torch.rand(3), requires_grad = True)
        self.dif4 = torch.nn.Parameter(torch.rand(3), requires_grad = True)
        self.dif5 = torch.nn.Parameter(torch.rand(3), requires_grad = True)
        self.dif6 = torch.nn.Parameter(torch.rand(3), requires_grad = True)


        self.dec_linear        = torch.nn.Linear(in_features= 6, out_features=1, bias=False)

        # power_allocation weights
        self.weight_all = torch.nn.Parameter(torch.ones(args.code_rate),requires_grad = True)
        self.weight_first_4 = torch.nn.Parameter(torch.ones(4),requires_grad = True)
        self.weight_last_5 = torch.nn.Parameter(torch.ones(5),requires_grad = True)
    
    def normalize(self, data):
        if self.args.batch_norm == True:
            batch_mean = torch.mean(data, 0)
            batch_std = torch.std(data,0)

        else:
            # get a precalculated norm
            id = str(self.args.feedback_SNR)+'_'+str(self.args.forward_SNR)+'_'+str(self.args.enc_num_unit)
            with open('meanvar_py/meanvar_'+id+'.pkl', 'rb') as file:  # Python 3: open(..., 'wb')
                loaded_data = pickle.load(file)
                batch_mean, batch_std = loaded_data['mean'], loaded_data['var']

        data_normalized = (data - batch_mean)*1.0/batch_std

        return data_normalized

    def power_allocation(self, data):
        
        data[:,:,0] = torch.multiply( data[:,:,0].clone(), self.weight_all[0])
        data[:,:,1] = torch.multiply( data[:,:,1].clone(), self.weight_all[1])
        data[:,:,2] = torch.multiply( data[:,:,2].clone(), self.weight_all[2])

        for idx_bit in range(4):
            data[:,idx_bit,0] = torch.multiply( data[:,idx_bit,0].clone(), self.weight_first_4[idx_bit])
            data[:,idx_bit,1] = torch.multiply( data[:,idx_bit,1].clone(), self.weight_first_4[idx_bit])
            data[:,idx_bit,2] = torch.multiply( data[:,idx_bit,2].clone(), self.weight_first_4[idx_bit])
        
        idx_start = self.args.block_len+1 - 5
        for idx_bit in range(5):
            data[:,idx_start+idx_bit,0] = torch.multiply( data[:,idx_start+idx_bit,0].clone(), self.weight_last_5[idx_bit])
            data[:,idx_start+idx_bit,1] = torch.multiply( data[:,idx_start+idx_bit,1].clone(), self.weight_last_5[idx_bit])
            data[:,idx_start+idx_bit,2] = torch.multiply( data[:,idx_start+idx_bit,2].clone(), self.weight_last_5[idx_bit])
            
        rem = self.args.block_len+1 - 4 - 5
        den = (rem + sum(self.weight_first_4**2) + sum(self.weight_last_5**2)) * sum(self.weight_all**2)
        power_allocation_output = torch.multiply( torch.sqrt(self.args.code_rate * (self.args.block_len+1) / den ), data)

        return power_allocation_output 

    def forward(self, information_bits, forward_noise, feedback_noise):

        num_samples_input = information_bits.shape[0]
        block_len = information_bits.shape[1]

        # encoder part: Phase 1

        codewords_phase_1 = 2.0 * information_bits-1.0

        # encoder part: Phase 2

        # nonrecurrent part
        def parity1(b, n1):
            """
            n1 represents the noises in phase 1 
            b_i
            n_i
            """
            offset = torch.where(b == 0, self.e[3], - self.e[4])
            indicator = ((- (2 * b - 1) * (n1 + offset)) > 0).to(torch.int)
            p1 = self.e[0] * (n1 + offset) * indicator
            return p1
        
        def parity2(b,n1):
            """
            n1 represents the noises in phase 1 
            b_i
            n_i
            """
            offset = torch.where(b == 0, self.e[3], - self.e[4])
            indicator = ((- (2 * b - 1) * (n1 + offset)) > 0).to(torch.int)
            p2 = - self.e[0] * (n1 + offset) * indicator
            return p2


        # recurrent part
        def h4(n2, n3, b, n1):
            """
            n2 represents the noises in parity 1
            n3 represents the noises in parity 2
            n2_{i-1}
            n3_{i-1}
            b_{i-1}
            n1_{i-1}
            """
            condition = (b == 0) 
            branch1 = torch.tanh(- self.k[0] * n1 + self.k[1] * n2 - self.k[2] * n3 + self.k[3])
            dim1, dim2 = branch1.shape
            branch2 = torch.ones(dim1, dim2).to(device)
            res = torch.where(condition, branch1, branch2)
            return res
            
        def h5(n2, n3, b, n1):
            """
            n2_{i-1}
            n3_{i-1}
            b_{i-1}
            n1_{i-1}
            """
            condition = (b == 1) 
            branch1 = torch.tanh(- self.k[0] * n1 + self.k[1] * n2 - self.k[2] * n3 - self.k[3])
            dim1, dim2 = branch1.shape
            branch2 = -1 * torch.ones(dim1, dim2).to(device)
            res = torch.where(condition, branch1, branch2)
            return res

        def h6(n2, n3, h4, h7):
            """
            n2 represents the noises in parity 1
            n3 represents the noises in parity 2
            n2_{i-1}
            n3_{i-1}
            h4_{i-1}
            h7_{i-1}
            """
            res = torch.tanh(self.m[0] * n2 + self.m[1] * n3 + self.m[2] * h4 + self.m[3] * h7 + self.m[4])
            return res
            
        def h7(n2, n3, h5, h6):
            """
            n2_{i-1}
            n3_{i-1}
            h5_{i-1}
            h6_{i-1}
            """
            res = torch.tanh(- self.m[0] * n2 - self.m[1] * n3 - self.m[2] * h5 + self.m[3] * h6 + self.m[4])

            return res

        all_h4 = []
        all_h5 = []
        all_h6 = []
        all_h7 = []
        for idx_bit in range(information_bits.shape[1]):
            
            if idx_bit == 0:
                noise_tmp = forward_noise[:,idx_bit,:] + feedback_noise[:,idx_bit,:]
                noise_tmp[:,1:] = torch.zeros(noise_tmp[:,1:].shape)

                input_tmp           =   torch.cat([information_bits[:,idx_bit,:].view(num_samples_input, 1, 1),
                                                   noise_tmp.view(num_samples_input, 1, 3)], dim=2)
                p1_linear = parity1(input_tmp[:,:,0], input_tmp[:,:,1])
                p2_linear = parity2(input_tmp[:,:,0], input_tmp[:,:,1])
                vh4 = torch.ones(num_samples_input, 1).to(device)
                vh5 = -1 * torch.ones(num_samples_input, 1).to(device)

                vh6 = torch.ones(num_samples_input, 1).to(device)
                vh7 = torch.ones(num_samples_input, 1).to(device)
                all_h4.append(vh4)
                all_h5.append(vh5)
                all_h6.append(vh6)
                all_h7.append(vh7)

            else:
                noise_tmp = forward_noise[:,idx_bit,:] + feedback_noise[:,idx_bit,:]
                noise_tmp[:,1:] = forward_noise[:,idx_bit-1,1:] + feedback_noise[:,idx_bit-1,1:]

                input_tmp           =   torch.cat([information_bits[:,idx_bit,:].view(num_samples_input, 1, 1),
                                                   noise_tmp.view(num_samples_input, 1, 3)], dim=2)
                p1_linear = parity1(input_tmp[:,:,0], input_tmp[:,:,1])
                p2_linear = parity2(input_tmp[:,:,0], input_tmp[:,:,1])

                past_info = information_bits[:,idx_bit-1,:].view(num_samples_input, 1, 1)
                past_n1 = (forward_noise[:,idx_bit-1,0]+ feedback_noise[:,idx_bit-1,0]).view(num_samples_input, 1, 1)
                vh4 = h4(input_tmp[:,:,2], input_tmp[:,:,3], past_info[:,:,0], past_n1[:,:,0])
                vh5 = h5(input_tmp[:,:,2], input_tmp[:,:,3], past_info[:,:,0], past_n1[:,:,0])

                vh6 = h6(input_tmp[:,:,2], input_tmp[:,:,3], all_h4[idx_bit-1], all_h7[idx_bit-1])
                vh7 = h7(input_tmp[:,:,2], input_tmp[:,:,3], all_h5[idx_bit-1], all_h6[idx_bit-1])

                all_h4.append(vh4)
                all_h5.append(vh5)
                all_h6.append(vh6)
                all_h7.append(vh7)

            # parity bits
            p1 = p1_linear - self.e[1] * vh4 - self.e[1] * vh5 - self.e[2] * vh6 + self.e[2] * vh7
            p2 = p2_linear - self.e[1] * vh4 - self.e[1] * vh5 - self.e[2] * vh6 + self.e[2] * vh7
            
            linear_output = torch.cat([p1.unsqueeze(-1), p2.unsqueeze(-1)], axis=2)
        
            if idx_bit == 0:
                codewords_phase_2= linear_output.view(num_samples_input, 1, 2)
            else:
                codewords_phase_2 = torch.cat([codewords_phase_2,linear_output], dim = 1)
        
        norm_output  = self.normalize(codewords_phase_2)
        
        cat_codewords = torch.cat([codewords_phase_1, norm_output], axis=2)

        #power allocation 
        power_allocation_output = self.power_allocation(cat_codewords)
        codewords = power_allocation_output
    
        # AWGN channel
        noisy_codewords = codewords + forward_noise

        # decoder
        noisy_yi = noisy_codewords[:,:,0].view(num_samples_input, block_len, 1)
        noisy_yi1 = noisy_codewords[:,:,1].view(num_samples_input, block_len, 1)
        noisy_yi2 = noisy_codewords[:,:,2].view(num_samples_input, block_len, 1)

        # first-stage
        g1 = torch.tanh(self.d1[0] * noisy_yi - self.d1[1] * noisy_yi1 + self.d1[1] * noisy_yi2)
        g2 = torch.tanh(self.d2[0] * noisy_yi - self.d2[1] * noisy_yi1 + self.d2[1] * noisy_yi2)
        g3 = torch.tanh(self.d3[0] * noisy_yi - self.d3[1] * noisy_yi1 + self.d3[1] * noisy_yi2)

        current_parity_sum = noisy_yi1 + noisy_yi2
        onefuture_parity_sum = torch.cat((current_parity_sum[:,1:,:], torch.zeros(num_samples_input, 1, 1)), dim=1)
        twofuture_parity_sum = torch.cat((current_parity_sum[:,2:,:], torch.zeros(num_samples_input, 2, 1)), dim=1)
        threefuture_parity_sum = torch.cat((current_parity_sum[:,3:,:], torch.zeros(num_samples_input, 3, 1)), dim=1)

        phase1_output = torch.cat([g1, g2, g3, onefuture_parity_sum, twofuture_parity_sum, threefuture_parity_sum], dim=2)

        # second-stage
        # forward direction
        fwd1 = torch.tanh(self.fw1[0] * g1 + self.fw1[1] * g2 + self.fw1[2] * g3 - self.fw1[3] * onefuture_parity_sum - self.fw1[4] * twofuture_parity_sum - self.fw1[5] * threefuture_parity_sum)
        fwd2 = torch.tanh(self.fw2[0] * g1 + self.fw2[1] * g2 + self.fw2[2] * g3 - self.fw2[3] * onefuture_parity_sum - self.fw2[4] * twofuture_parity_sum - self.fw2[5] * threefuture_parity_sum)
        fwd3 = torch.tanh(self.fw3[0] * g1 + self.fw3[1] * g2 + self.fw3[2] * g3 - self.fw3[3] * onefuture_parity_sum - self.fw3[4] * twofuture_parity_sum - self.fw3[5] * threefuture_parity_sum)

        diff1 = self.dif1[0] * fwd1 + self.dif1[1] *fwd2 + self.dif1[2] *fwd3
        diff2 = self.dif2[0] * fwd1 + self.dif2[1] *fwd2 + self.dif2[2] *fwd3
        diff3 = self.dif3[0] * fwd1 + self.dif3[1] *fwd2 + self.dif3[2] *fwd3
        fw_diff1 = torch.cat((torch.zeros(num_samples_input, 1, 1), diff1[:,:-1,:]), dim=1)
        fw_diff2 = torch.cat((torch.zeros(num_samples_input, 1, 1), diff2[:,:-1,:]), dim=1)
        fw_diff3 = torch.cat((torch.zeros(num_samples_input, 1, 1), diff3[:,:-1,:]), dim=1)

        new_fwd1 = torch.tanh(self.fw1[0] * g1 + self.fw1[1] * g2 + self.fw1[2] * g3 - self.fw1[3] * onefuture_parity_sum - self.fw1[4] * twofuture_parity_sum - self.fw1[5] * threefuture_parity_sum + fw_diff1)
        new_fwd2 = torch.tanh(self.fw2[0] * g1 + self.fw2[1] * g2 + self.fw2[2] * g3 - self.fw2[3] * onefuture_parity_sum - self.fw2[4] * twofuture_parity_sum - self.fw2[5] * threefuture_parity_sum + fw_diff2)
        new_fwd3 = torch.tanh(self.fw3[0] * g1 + self.fw3[1] * g2 + self.fw3[2] * g3 - self.fw3[3] * onefuture_parity_sum - self.fw3[4] * twofuture_parity_sum - self.fw3[5] * threefuture_parity_sum + fw_diff3)

        # backward direction
        back1 = torch.tanh(self.bc1[0] * g1 + self.bc1[1] * g2 + self.bc1[2] * g3 - self.bc1[3] * onefuture_parity_sum - self.bc1[4] * twofuture_parity_sum - self.bc1[5] * threefuture_parity_sum)
        back2 = torch.tanh(self.bc2[0] * g1 + self.bc2[1] * g2 + self.bc2[2] * g3 - self.bc2[3] * onefuture_parity_sum - self.bc2[4] * twofuture_parity_sum - self.bc2[5] * threefuture_parity_sum)
        back3 = torch.tanh(self.bc3[0] * g1 + self.bc3[1] * g2 + self.bc3[2] * g3 - self.bc3[3] * onefuture_parity_sum - self.bc3[4] * twofuture_parity_sum - self.bc3[5] * threefuture_parity_sum)

        diff1b = self.dif4[0] * back1 + self.dif4[1] * back2 + self.dif4[2] * back3
        diff2b = self.dif5[0] * back1 + self.dif5[1] * back2 + self.dif5[2] * back3
        diff3b = self.dif6[0] * back1 + self.dif6[1] * back2 + self.dif6[2] * back3

        bc_diff1 = torch.cat((diff1b[:,1:,:], torch.zeros(num_samples_input, 1, 1)), dim=1)
        bc_diff2 = torch.cat((diff2b[:,1:,:], torch.zeros(num_samples_input, 1, 1)), dim=1)
        bc_diff3 = torch.cat((diff3b[:,1:,:], torch.zeros(num_samples_input, 1, 1)), dim=1)

        new_back1 = torch.tanh(self.bc1[0] * g1 + self.bc1[1] * g2 + self.bc1[2] * g3 - self.bc1[3] * onefuture_parity_sum - self.bc1[4] * twofuture_parity_sum - self.bc1[5] * threefuture_parity_sum +  bc_diff1)
        new_back2 = torch.tanh(self.bc2[0] * g1 + self.bc2[1] * g2 + self.bc2[2] * g3 - self.bc2[3] * onefuture_parity_sum - self.bc2[4] * twofuture_parity_sum - self.bc2[5] * threefuture_parity_sum +  bc_diff2)
        new_back3 = torch.tanh(self.bc3[0] * g1 + self.bc3[1] * g2 + self.bc3[2] * g3 - self.bc3[3] * onefuture_parity_sum - self.bc3[4] * twofuture_parity_sum - self.bc3[5] * threefuture_parity_sum +  bc_diff3)

        phase2_output = torch.cat((new_fwd1, new_fwd2,new_fwd3,new_back1, new_back2, new_back3), dim=2)
        decoder_output     = torch.sigmoid(self.dec_linear( phase2_output))

        all_hidden_states = []
        return all_hidden_states, codewords, decoder_output

    

args = get_args(jupyter_notebook = True)

print('args = ', args.__dict__)

if args.feedback_SNR == 100:
    feedback_sigma = 0
else:
    feedback_sigma = snr_db_2_sigma(args.feedback_SNR)

forward_sigma = snr_db_2_sigma(args.forward_SNR)

use_cuda = args.with_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('use_cuda = ', use_cuda)
print('device = ', device)

if use_cuda:
    model = AE(args).to(device)
else:
    model = AE(args)
print(model)

args.initial_weights = 'logs/noisy/enc3dec4_vary_fbminus3.pt'
if args.initial_weights == 'default':
    pass
elif args.initial_weights == 'deepcode':
    f_load_deepcode_weights(model)
    print('deepcode weights are loaded.')
else:
    model.load_state_dict(torch.load(args.initial_weights, map_location=torch.device('cpu')))
    model.args = args
    print('initial weights are loaded.')


optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate, betas=(0.9,0.999), eps=1e-07, weight_decay=0, amsgrad=False)
learning_rate_step_size = int(10**6 / args.batch_size)
print('learning_rate_step_size = ', learning_rate_step_size)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_step_size, gamma=0.1)

X_validation    = torch.randint(0, 2, (args.num_samples_validation, args.block_len, 1))
X_validation = torch.cat([X_validation, torch.zeros(args.num_samples_validation, 1, 1)], dim=1)
forward_noise_validation = forward_sigma * torch.randn((args.num_samples_validation, args.block_len+1, args.code_rate))
feedback_noise_validation   = feedback_sigma * torch.randn((args.num_samples_validation, args.block_len+1, args.code_rate))
# torch.zeros((args.num_samples_validation, args.block_len+1, args.code_rate))  # perfect feedback

X_validation, forward_noise_validation, feedback_noise_validation = X_validation.to(device), forward_noise_validation.to(device), feedback_noise_validation.to(device)

loss_his, ber_his, bler_his, rnn_output_his, codewords_his, decoder_output_his = validation(model, device, X_validation, forward_noise_validation, feedback_noise_validation)

print('----- Validation BER (initial): ', ber_his)
print('----- Validation loss (initial): ', loss_his)



###training part
# writer = SummaryWriter(log_dir = './logs/deepcode/model_'+date.today().strftime("%Y%m%d")+'_'+identity)

# for epoch in range(1, args.num_epoch + 1):
#     loss_train, X_train, forward_noise_train, feedback_noise_train, rnn_output, codewords, decoder_output = train(args, model, device, optimizer, scheduler)
#     print('--- epoch {} with training loss = {}'.format(epoch, loss_train))
#     writer.add_scalar('loss_train', loss_train, epoch)

#     if epoch%10 == 0:
#         loss_validation, ber_validation, bler_validation,  rnn_output_validation, codewords_validation, decoder_output_validation = validation(model, device, X_validation, forward_noise_validation, feedback_noise_validation)
#         writer.add_scalar('loss_validation', loss_validation, epoch)
#         print('----- Validation BER: ', ber_validation)
#         print('----- Validation BLER: ', bler_validation)
#         print('----- Validation loss: ', loss_validation)

#         #if ber_validation < ber_his:
#         ber_his = ber_validation
#         file_name = './logs/deepcode/model_'+date.today().strftime("%Y%m%d")+'_'+identity+'.pt'
#         torch.save(model.state_dict(), file_name)
#         print('saved model as file: ', file_name)

#         file_name = './logs/deepcode/model_'+date.today().strftime("%Y%m%d")+'_'+identity+'_args.json'
#         json_object = dumps(args.__dict__)
#         with open(file_name, "w") as open_file:
#             open_file.write(json_object)

# print('final saved file_name = ', file_name)

        
