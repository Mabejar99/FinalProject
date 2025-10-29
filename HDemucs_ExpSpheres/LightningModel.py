import torch
import lightning as pl
from torchmetrics.audio import SignalDistortionRatio
import torch.nn.functional as F
from LossFunction import perceptual_loss
from lightning.pytorch.utilities import grad_norm
import auraloss

class LightningModel(pl.LightningModule):
    def __init__(self, model, optimizer, conf):
        super(LightningModel, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer = optimizer
        self.conf = conf

        #self.loss_fn = m2m_sdr_loss
        self.shuller = perceptual_loss
        self.sdr = SignalDistortionRatio()
        self.auralossnew = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            scale="mel",
            n_bins=128,
            sample_rate=model.samplerate,
            perceptual_weighting=True,
        )


    def forward(self, batched_mixtures):
        print("\n[FORWARD] --- INPUT TO MODEL ---")
        print(f"Shape: {batched_mixtures.shape}")
        print(f"Min: {batched_mixtures.min().item():.4f}, Max: {batched_mixtures.max().item():.4f}")
        print(f"NaNs present: {torch.isnan(batched_mixtures).any().item()}")
        print(f"Infs present: {torch.isinf(batched_mixtures).any().item()}")
        # Asegura que es [B, C, T] antes de meterlo al modelo
        if batched_mixtures.shape[1] > batched_mixtures.shape[2]:
         # Está en [B, T, C] → hay que permutar
            batched_mixtures = batched_mixtures.permute(0, 2, 1)
            print(f"Permuted shape: {batched_mixtures.shape} (now [B, C, T])")
        estimated_sources = self.model(batched_mixtures)
        print("[FORWARD] --- OUTPUT FROM MODEL ---")
        print(f"Shape: {estimated_sources.shape}")
        print(f"Min: {estimated_sources.min().item():.4f}, Max: {estimated_sources.max().item():.4f}")
        print(f"NaNs present: {torch.isnan(estimated_sources).any().item()}")
        print(f"Infs present: {torch.isinf(estimated_sources).any().item()}")
        print("")
        return estimated_sources


    def training_step(self, batch, batch_idx):
        # unzip batch data
        mix, refs,instrument_names = batch
        if batch_idx == 0:
            print("Instrumentos en batch:", instrument_names)
        
        assert not torch.isnan(mix).any(), (
            f'{torch.isnan(mix).sum().item()} NaN values in input mixtures!'
        )

        # training config
        permute_sources = self.conf.get('train').get('permute_sources')
        name_loss = self.conf.get('train').get('loss_function')

        # ramdom permutation of sources in each batch
        if permute_sources:
            B, S = mix.shape[0:2]
            for i in range(B):
                perm = torch.randperm(S)
                mix[i] = mix[i, perm]
                refs[i] = refs[i, perm]

        # model forward pass
        estimated_sources = self(mix)
        assert not torch.isnan(estimated_sources).any(), (
            f'{torch.isnan(estimated_sources).sum().item()} NaN values in estimation!'
        )

        # compute the M2M SDR/L1 loss
        # train_sdr_loss = self.loss_fn(
        #                             estimated_sources,
        #                             mix,
        #                             mic_transfer_fn
        #                         )

        # compute the S2S L1 loss
        if name_loss == 'l1':
            loss = F.l1_loss(estimated_sources, refs, reduction='none').mean()
            self.log('train_L1_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # compute the S2S NEW auraloss (Yamamoto + perceptual)
        if name_loss == 'auraloss':
            estimated_sources = estimated_sources.contiguous()
            refs = refs.contiguous()
            loss = self.auralossnew(estimated_sources, refs)
            self.log('train_AuralossNEW_loss', loss,
                    prog_bar=True, on_epoch=True, sync_dist=True)

        # compute the S2S perceptual loss (Schuller)
        if name_loss == 'perceptual':
            loss = self.shuller(estimated_sources, refs, self.model.samplerate)
            self.log('train_Perceptual_loss', loss,
                    prog_bar=True, on_epoch=True, sync_dist=True)

        # compute the S2S SDR loss
        if name_loss == 'sdr':
            try:
                loss = -self.sdr(estimated_sources, refs)
            except torch.linalg.LinAlgError:
                loss = -self.sdr(estimated_sources, refs + torch.finfo(float).eps)
            self.log('train_sdr_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # compute S2S SDR training set (NOT LOSS)
        try:
            train_sdr = self.sdr(estimated_sources, refs)
        except torch.linalg.LinAlgError:
            train_sdr = self.sdr(estimated_sources, refs + torch.finfo(float).eps)
        self.log('train_sdr', train_sdr,
                 prog_bar=True, on_epoch=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # dataloader_idx log labels
        log_label_sdr = [
            'same_acoustics_same_layout_val_sdr',
            'diff_acoustics_same_layout_val_sdr',
            'same_acoustics_diff_layout_val_sdr',
            'diff_acoustics_diff_layout_val_sdr'][dataloader_idx]
        log_label_auralossnew = [
            'same_acoustics_same_layout_val_AuralossNew',
            'diff_acoustics_same_layout_val_AuralossNew',
            'same_acoustics_diff_layout_val_AuralossNew',
            'diff_acoustics_diff_layout_val_AuralossNew'][dataloader_idx]
        log_label_perceptual = [
            'same_acoustics_same_layout_val_Perceptual',
            'diff_acoustics_same_layout_val_Perceptual',
            'same_acoustics_diff_layout_val_Perceptual',
            'diff_acoustics_diff_layout_val_Perceptual'][dataloader_idx]

        # unzip batch data
        mix, refs,_ = batch
        #refs = refs.permute(0,2,1)  # [B,T,S] -> [B,S,T]
        #mix = mix.permute(0,2,1)    # [B,T,C] -> [B,C,T]
        assert not torch.isnan(mix).any(), (
            f'{torch.isnan(mix).sum().item()} NaN values in input mixtures!'
        )

        # training config
        permute_sources = self.conf.get('train').get('permute_sources')

        # ramdom permutation of sources in each batch
        if permute_sources:
            B, S = mix.shape[0:2]
            for i in range(B):
                perm = torch.randperm(S)
                mix[i] = mix[i, perm]
                refs[i] = refs[i, perm]

        # model forward pass
        estimated_sources = self(mix)
        assert not torch.isnan(estimated_sources).any(), (
            f'{torch.isnan(estimated_sources).sum().item()} NaN values estimated sources!'
        )

        # compute the SDR
        try:
            val_sdr = self.sdr(estimated_sources, refs)
        except torch.linalg.LinAlgError:
            val_sdr = self.sdr(estimated_sources, refs + torch.finfo(float).eps)
        self.log(log_label_sdr, val_sdr,
                 prog_bar=True, on_epoch=True, sync_dist=True)

        # compute the NEW auraloss (Yamamoto + perceptual)
        # estimated_sources = estimated_sources.contiguous()
        # refs = refs.contiguous()
        # val_auralossnew = self.auralossnew(estimated_sources, refs)
        # self.log(log_label_auralossnew, val_auralossnew,
        #          prog_bar=True, on_epoch=True, sync_dist=True)

        # compute the S2S perceptual loss (Schuller)
        # val_perceptual = self.shuller(estimated_sources, refs, self.model.samplerate)
        # self.log(log_label_perceptual, val_perceptual,
        #          prog_bar=True, on_epoch=True, sync_dist=True)

        return val_sdr


    def configure_optimizers(self):
        if self.optimizer:
            return self.optimizer
        else:
            # Default optimizer if none is passed
            return torch.optim.Adam(self.parameters(), lr=1e-3)


    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision,
        # the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)
