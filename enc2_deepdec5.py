# enc 2 + deep decoder with 5 hidden states
# noiseless feedback
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import date
from json import dumps
import itertools

# torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

identity = str(np.random.random())[2:8]
identity = 'enc2_deepdec5'
print('[ID]', identity)

torch.cuda.empty_cache()

def get_args(jupyter_notebook):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-code_rate', type=int, default=3)
    parser.add_argument('-block_len', type=int, default=50, help='This do not including zero-padding')
    parser.add_argument('-num_samples_train', type=int, default=60000)
    parser.add_argument('-num_samples_validation', type=int, default=200000) # 2000000
    parser.add_argument('-num_samples_test', type=int, default=200)

    parser.add_argument('-dec_num_unit', type=int, default=5)

    parser.add_argument('-feedback_SNR', type=int, default=100, help='100 means noiseless feeback')
    parser.add_argument('-forward_SNR', type=int, default=0)

    parser.add_argument('-batch_size', type=int, default=300) 

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

    comparisin_result = torch.ne(torch.round(y_true), torch.round(y_pred)).float()  # how many different bits
    if positions == 'default':
        res = torch.sum(comparisin_result)/(comparisin_result.shape[0]*comparisin_result.shape[1])  # new
    else:   
        res = torch.mean(comparisin_result, dim=0).type(torch.FloatTensor)
        for pos in positions:
            res[pos] = 0.0
        res = torch.mean(res)
    return res

    # t1 = torch.round(y_true[:,:,:])
    # t2 = torch.round(y_pred[:,:,:])

    # myOtherTensor = np.not_equal(t1, t2).float()
    # k = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    # return k

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
        self.e = torch.nn.Parameter(torch.rand(2), requires_grad = True)
        self.k = torch.nn.Parameter(torch.rand(4), requires_grad = True)

        # Decoder

        self.dec_gru_1           = torch.nn.GRU(input_size=args.code_rate,  hidden_size=args.dec_num_unit, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.dec_norm_1          = torch.nn.BatchNorm1d(num_features=2*args.dec_num_unit, eps=0.001, momentum=0.01)

        self.dec_gru_2           = torch.nn.GRU(input_size=2*args.dec_num_unit,  hidden_size=args.dec_num_unit, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.dec_norm_2          = torch.nn.BatchNorm1d(num_features=2*args.dec_num_unit, eps=0.001, momentum=0.01)


        self.dec_linear        = torch.nn.Linear(in_features=2*args.dec_num_unit, out_features=1, bias=True)

        # power_allocation weights
        self.weight_all = torch.nn.Parameter(torch.ones(args.code_rate),requires_grad = True)
        self.weight_first_4 = torch.nn.Parameter(torch.ones(4),requires_grad =True)
        self.weight_last_5 = torch.nn.Parameter(torch.ones(5),requires_grad = True)
    
    def normalize(self, data):
        if self.args.batch_norm == True:
            batch_mean = torch.mean(data, 0)
            batch_std = torch.std(data,0)
            data_normalized = (data - batch_mean)*1.0/batch_std
            if False:
                print('normalize with means = ', batch_mean)
                print('normalize with stds = ', batch_std)
        else:
            data_normalized = data # todo - get a precalculated norm
        
        return data_normalized
    
    def power_allocation(self, data):
        
        data[:,:,0] = torch.multiply( data[:,:,0].clone(), self.weight_all[0])
        data[:,:,1] = torch.multiply( data[:,:,1].clone(), self.weight_all[1])
        data[:,:,2] = torch.multiply( data[:,:,2].clone(), self.weight_all[2])

        for idx_bit in range(4):
            data[:,idx_bit,0] = torch.multiply( data[:,idx_bit,0].clone(), self.weight_first_4[idx_bit])
            data[:,idx_bit,1] = torch.multiply( data[:,idx_bit,1].clone(), self.weight_first_4[idx_bit])
            data[:,idx_bit,2] = torch.multiply( data[:,idx_bit,2].clone(), self.weight_first_4[idx_bit])
        
        idx_start = self.args.block_len+1 -1 - 5 + 1
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

        # encoder part: Phase 1

        codewords_phase_1 = 2.0*information_bits-1.0

        # encoder part: Phase 2

        # nonrecurrent part
        def parity1(b, n1):
            """
            n1 represents the noises in phase 1 
            b_i
            n_i
            """
            indictor = ((- (2 * b - 1) * n1) > 0).to(torch.int)
            p1 = self.e[0] * n1 * indictor
            return p1
            #.unsqueeze(-1)
        
        def parity2(b,n1):
            """
            n1 represents the noises in phase 1 
            b_i
            n_i
            """
            indictor = ((- (2 * b - 1) * n1) > 0).to(torch.int)
            p2 = - self.e[0] * n1 * indictor
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

            else:
                noise_tmp = forward_noise[:,idx_bit,:] + feedback_noise[:,idx_bit,:]
                noise_tmp[:,1:] = forward_noise[:,idx_bit-1,1:] + feedback_noise[:,idx_bit-1,1:]

                input_tmp           =   torch.cat([information_bits[:,idx_bit,:].view(num_samples_input, 1, 1),
                                                   noise_tmp.view(num_samples_input, 1, 3)], dim=2)
                p1_linear = parity1(input_tmp[:,:,0], input_tmp[:,:,1])
                p2_linear = parity2(input_tmp[:,:,0], input_tmp[:,:,1])

                past_info = information_bits[:,idx_bit-1,:].view(num_samples_input, 1, 1)
                past_n1 = forward_noise[:,idx_bit-1,0].view(num_samples_input, 1, 1)
                vh4 = h4(input_tmp[:,:,2], input_tmp[:,:,3], past_info[:,:,0], past_n1[:,:,0])
                vh5 = h5(input_tmp[:,:,2], input_tmp[:,:,3], past_info[:,:,0], past_n1[:,:,0])
 

            # parity bits
            p1 = p1_linear - self.e[1] * vh4 - self.e[1] * vh5
            
            p2 = p2_linear - self.e[1] * vh4 - self.e[1] * vh5
            

            linear_output = torch.cat([p1.unsqueeze(-1), p2.unsqueeze(-1)], axis=2)
        
            if idx_bit == 0:
                codewords_phase_2= linear_output.view(num_samples_input, 1, 2)
            else:
                codewords_phase_2 = torch.cat([codewords_phase_2,linear_output], dim = 1)
        
        norm_output  = self.normalize(codewords_phase_2)
        
        cat_codewords = torch.cat([codewords_phase_1, norm_output], axis=2)

        ####power allocation part#####
        power_allocation_output = self.power_allocation(cat_codewords)
        codewords = power_allocation_output

        # AWGN channel
        noisy_codewords = codewords + forward_noise

        # decoder

        gru_output_1, _  = self.dec_gru_1(noisy_codewords)
        gru_output_1 = torch.transpose(gru_output_1, 1,2)

        dec_norm_output_1 = self.dec_norm_1(gru_output_1)
        dec_norm_output_1 = torch.transpose(dec_norm_output_1, 1,2)

        gru_output_2, _  = self.dec_gru_2(dec_norm_output_1)
        gru_output_2 = torch.transpose(gru_output_2, 1,2)
        
        dec_norm_output_2 = self.dec_norm_2(gru_output_2)
        dec_norm_output_2 = torch.transpose(dec_norm_output_2, 1,2)
        decoder_output     = torch.sigmoid(self.dec_linear(dec_norm_output_2))

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

args.initial_weights = 'logs/noiseless/enc2_deepdec5_snr0.pt'
if args.initial_weights == 'default':
    pass
elif args.initial_weights == 'deepcode':
    f_load_deepcode_weights(model)
    print('deepcode weights are loaded.')
else:
    model.load_state_dict(torch.load(args.initial_weights), strict = False)
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

X_validation, forward_noise_validation, feedback_noise_validation = X_validation.to(device), forward_noise_validation.to(device), feedback_noise_validation.to(device)

loss_his, ber_his, bler_his, rnn_output_his, codewords_his, decoder_output_his = validation(model, device, X_validation, forward_noise_validation, feedback_noise_validation)

print('----- Validation BER (initial): ', ber_his)
print('----- Validation loss (initial): ', loss_his)


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
# torch.cuda.empty_cache()