import pathlib
import torch
import numpy as np
from torchaudio.functional import lfilter
from torchmetrics.audio import SignalDistortionRatio
import torch.nn.functional as F
from pythonaudiocoder.psyacloss_torch import percloss


rir_filepath = pathlib.Path(__file__).parent.resolve() /'RIR' / 'Strings.npy'

def circ_conv(x,h):
    N = max([len(x),len(h)])

    Y = torch.fft.fft(x,N)*torch.fft.fft(h,N)
    y = torch.real(torch.fft.ifft(Y))

    return y

def m2m_sdr_loss(estimates, mixture, mic_transfer_fn):
    '''
    estimates shape [batch_size,n_srcs,n_samples]
    mixture shape [batch_size,n_mics,n_samples]
    mic_transfer_fn [batch_size,n_sources,n_n_mics,n_samples]
    '''
    assert (estimates.shape[-1] > estimates.shape[1]), f'estimates shape must be (batch_size,n_src,n_samples), but is {tuple(estimates.shape)}'
    assert (mixture.shape[-1] > mixture.shape[1]), f'mixture shape must be (batch_size,n_mics,n_samples), but is {tuple(estimates.shape)}'
    # cast all the tensors into the same datatype if not the same
    if not ((estimates.dtype == mixture.dtype) and (estimates.dtype == mic_transfer_fn.dtype)):
        mixture.type(estimates.dtype)
        mic_transfer_fn.type(estimates.dtype)
    #obtain dimensional information
    batch_size = estimates.shape[0]
    n_mics = mixture.shape[1]
    estimates_len = estimates.shape[-1]
    mic_transfer_fn_len = mic_transfer_fn.shape[-1]
    # compute the start sample where the linear and circular convolution are the same
    start = np.min([estimates_len,mic_transfer_fn_len]) - 1
    # compute the length of the valid convolution
    N = np.max([estimates_len,mic_transfer_fn_len])
    valid_length = N - start
    # initialize tensor to allocate the estimated mixtures
    mixture_hat = torch.zeros([batch_size,n_mics,valid_length],dtype=estimates.dtype,device=estimates.device)
    # iterate over the batches
    for batch_idx in range(batch_size):
        # iterate over the different microphones
        for mic_idx in range(n_mics):
            # iterate over the different close mic source estimations             
            for source_idx in range(estimates.shape[1]):
                # compute the close microphone estimated mixture for each mic 
                mixture_hat[batch_idx,mic_idx,:] += circ_conv(
                                            estimates[batch_idx,source_idx,:],
                                            mic_transfer_fn[batch_idx,source_idx,mic_idx,:],
                                            )[start::]
    # trim the input mixture to match the length of the estimated
    mixture = mixture[:,:,start::]

    # compute loss between the input mixture and the estimated mixture

    # SDR
    # sdr = SignalDistortionRatio().to(estimates.device)
    # try:
    #     sdr_loss = -sdr(mixture_hat, mixture)
    # except torch.linalg.LinAlgError:
    #     sdr_loss = -sdr(mixture_hat, mixture + torch.finfo(float).eps)

    # L1
    sdr_loss = F.l1_loss(mixture_hat, mixture, reduction='none').mean()
    
    return sdr_loss

def m2m_loss_sdr(estimates,mixture,rir_filepath=rir_filepath):
    '''
    estimates shape [batch_size,n_srcs,n_samples]
    mixture shape [batch_size,n_mics,n_samples]
    RIR stack [n_spots,n_mics,n_samples]
    '''
    # load the RIR npy
    RIR = np.load(rir_filepath)

    assert (estimates.shape[1] == 4), f'estimates shape must be (batch_size,4,n_samples), but is {tuple(estimates.shape)}'
    assert (mixture.shape[1] == 5), f'input mixture shape must be (batch_size,5,n_samples), but is {tuple(mixture.shape)}'
    assert (RIR.shape[0:2] == (4,5)), f'RIR shape must be (4,5,n_samples), but are {tuple(RIR.shape)}'

    #obtain dimensional information
    batch_size = estimates.shape[0]
    n_mics = mixture.shape[1]
    rir_len = RIR.shape[2]
    estimates_len = estimates.shape[2]
    # compute the start sample where the linear and circular convolution are the same
    start = np.min([estimates_len,rir_len]) - 1
    # compute the length of the valid convolution
    N = np.max([estimates_len,rir_len])
    valid_length = N - start
    # convert to tensor of the same data type as estimates
    RIR = torch.from_numpy(RIR).type(estimates.dtype).to(estimates.device)
    # trim the input mixture to match the valid convolution length
    main_mic_mixture = mixture[:,0,start::]
    # compute the mixture in the main mic
    sdr_losses = torch.zeros(batch_size, dtype=estimates.dtype, device=estimates.device)
    for batch_idx in range(batch_size):
        # filter the estimates (presumably close mic signals) by their RIR
        mixture_hat = circ_conv(estimates[batch_idx,0,:],RIR[0,0,:])    # [n_spots,n_mics,n_samples]
        mixture_hat += circ_conv(estimates[batch_idx,1,:],RIR[1,0,:])    # [n_spots,n_mics,n_samples]
        mixture_hat += circ_conv(estimates[batch_idx,2,:],RIR[2,0,:])    # [n_spots,n_mics,n_samples]
        mixture_hat += circ_conv(estimates[batch_idx,3,:],RIR[3,0,:])    # [n_spots,n_mics,n_samples]
        # trim the estimated mixture to match the valid convolution length
        mixture_hat = mixture_hat[start::]
        # compute the negative sdr loss between the estimated main mic mixture and the actual mixture
        sdr = SignalDistortionRatio().to(estimates.device)
        try:
            sdr_loss = -sdr(mixture_hat,main_mic_mixture[batch_idx,:])
        except torch.linalg.LinAlgError:
            sdr_loss = -sdr(mixture_hat,main_mic_mixture[batch_idx,:] + torch.finfo(float).eps)
        # append to the sdr losses list
        sdr_losses[batch_idx] = sdr_loss
    # After iterating over the batches, compute the mean sdr loss
    loss = sdr_losses.mean()

    return loss

def m2m_loss_no_loop(estimates,mixture,rir_filepath=rir_filepath):
    '''
    estimates shape [batch_size,n_srcs,n_samples]
    mixture shape [batch_size,n_mics,n_samples]
    RIR stack [n_spots,n_mics,n_samples]
    '''
    # load the RIR npy
    RIR = np.load(rir_filepath)

    assert (estimates.shape[1] == 4), f'estimates shape must be (batch_size,4,n_samples), but is {tuple(estimates.shape)}'
    assert (mixture.shape[1] == 5), f'input mixture shape must be (batch_size,5,n_samples), but is {tuple(mixture.shape)}'
    assert (RIR.shape[0:2] == (4,5)), f'RIR shape must be (4,5,n_samples), but are {tuple(RIR.shape)}'

    #obtain dimensional information
    batch_size = estimates.shape[0]
    n_mics = mixture.shape[1]
    rir_len = RIR.shape[2]
    estimates_len = estimates.shape[2]
    # compute the start sample where the linear and circular convolution are the same
    start = np.min([estimates_len,rir_len]) - 1
    # compute the length of the valid convolution
    N = np.max([estimates_len,rir_len])
    valid_length = N - start
    # convert to tensor of the same data type as estimates
    RIR = torch.from_numpy(RIR).type(estimates.dtype).to(estimates.device)
    # trim the input mixture to match the valid convolution length
    mixture = mixture[:,:,start::]
    # compute the mixture in the main mic
    l1_losses = torch.zeros(batch_size, dtype=estimates.dtype, device=estimates.device)
    for batch_idx in range(batch_size):
        # compute the estimated main mic mixture. RIR shape [n_spots,n_mics,n_samples]
        main_mic_mixture_hat = circ_conv(estimates[batch_idx,0,:],RIR[0,0,:])     # violin source spot 2 violin mic rir
        main_mic_mixture_hat += circ_conv(estimates[batch_idx,1,:],RIR[1,0,:])    # viola source spot 2 violin mic rir
        main_mic_mixture_hat += circ_conv(estimates[batch_idx,2,:],RIR[2,0,:])    # cello source spot 2 violin mic rir
        main_mic_mixture_hat += circ_conv(estimates[batch_idx,3,:],RIR[3,0,:])    # bass source spot 2 violin mic rir
        # compute the estimated violin close mic mixture. RIR shape [n_spots,n_mics,n_samples]
        violin_mic_mixture_hat = circ_conv(estimates[batch_idx,0,:],RIR[0,1,:])     # violin source spot 2 violin mic rir
        violin_mic_mixture_hat += circ_conv(estimates[batch_idx,1,:],RIR[1,1,:])    # viola source spot 2 violin mic rir
        violin_mic_mixture_hat += circ_conv(estimates[batch_idx,2,:],RIR[2,1,:])    # cello source spot 2 violin mic rir
        violin_mic_mixture_hat += circ_conv(estimates[batch_idx,3,:],RIR[3,1,:])    # bass source spot 2 violin mic rir
        # compute the estimated viola close mic mixture
        viola_mic_mixture_hat = circ_conv(estimates[batch_idx,0,:],RIR[0,2,:])      # violin source spot 2 viola mic rir
        viola_mic_mixture_hat += circ_conv(estimates[batch_idx,1,:],RIR[1,2,:])     # viola source spot 2 viola mic rir
        viola_mic_mixture_hat += circ_conv(estimates[batch_idx,2,:],RIR[2,2,:])     # cello source spot 2 viola mic rir
        viola_mic_mixture_hat += circ_conv(estimates[batch_idx,3,:],RIR[3,2,:])     # bass source spot 2 viola mic rir
        # compute the estimated cello close mic mixture
        cello_mic_mixture_hat = circ_conv(estimates[batch_idx,0,:],RIR[0,3,:])    # [n_spots,n_mics,n_samples]
        cello_mic_mixture_hat += circ_conv(estimates[batch_idx,1,:],RIR[1,3,:])    # [n_spots,n_mics,n_samples]
        cello_mic_mixture_hat += circ_conv(estimates[batch_idx,2,:],RIR[2,3,:])    # [n_spots,n_mics,n_samples]
        cello_mic_mixture_hat += circ_conv(estimates[batch_idx,3,:],RIR[3,3,:])    # [n_spots,n_mics,n_samples]
        # compute the estimated bass close mic mixture
        bass_mic_mixture_hat = circ_conv(estimates[batch_idx,0,:],RIR[0,4,:])    # [n_spots,n_mics,n_samples]
        bass_mic_mixture_hat += circ_conv(estimates[batch_idx,1,:],RIR[1,4,:])    # [n_spots,n_mics,n_samples]
        bass_mic_mixture_hat += circ_conv(estimates[batch_idx,2,:],RIR[2,4,:])    # [n_spots,n_mics,n_samples]
        bass_mic_mixture_hat += circ_conv(estimates[batch_idx,3,:],RIR[3,4,:])    # [n_spots,n_mics,n_samples]
        # create the close_mixture_hat
        mixture_hat = torch.stack([main_mic_mixture_hat, violin_mic_mixture_hat, viola_mic_mixture_hat, cello_mic_mixture_hat, bass_mic_mixture_hat],dim=0)
        # trim the estimated mixture to match the valid convolution length
        mixture_hat = mixture_hat[:,start::]
        # compute the l1 loss between the estimated mixture and the actual mixture
        l1_loss = torch.nn.functional.l1_loss(mixture_hat,mixture[batch_idx,:,:])
        # append to the sdr losses list
        l1_losses[batch_idx] = l1_loss
    # After iterating over the batches, compute the mean sdr loss
    loss = l1_losses.mean()

    return loss


def perceptual_loss(estimates, refs, fs):
    B, C, _ = refs.shape
    total_loss = 0.0
    for b in range(B):
        for c in range(C):
            total_loss += percloss(refs[b, c, :], estimates[b, c, :], fs)
    return total_loss

