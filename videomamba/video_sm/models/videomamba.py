# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

########################################################################################################################
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}
########################################################################################################################


class Block(nn.Module):
    def __init__(
        self,
        dim, # Dimensionality of the input (and output) tensor
        mixer_cls, # Mixer layer
        norm_cls=nn.LayerNorm, # Normalization layer. Defaults to nn.LayerNorm.
        fused_add_norm=False, # Boolean indicating whether to fuse addition and normalization. It is related to the skip connections
        residual_in_fp32=False, # Residuals stored in 32-bit floating-point format. Defaults to False
        drop_path=0., # https://paperswithcode.com/method/droppath
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails" #RMSNorm stands for "Root Mean Square Layer Normalization"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            # Questo dovrebbe riferirsi alle skip connections + la layer normalization. In pratica si mettono insieme le due operazioni per migliorare l'efficienza computazionale. Se questo booleano fused_add_norm=False, si procede a fare l'operazione. 
            # Si sommano gli hidden states ai residui. Questi vengono poi normalizzati e diventano il nuovo hidden_states
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32: # questo riguarda solo la precision
                residual = residual.to(torch.float32)
        else:
            # Si normalizza, usando RMSNorm oppure layer_norm
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            # Si applica l'operazione
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

# Questa funzione è importante
def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
): 
    '''
    Funzione utilizzata per creare un'istanza di blocco della classe Block.
    Assegna anche un indice al blocco
    '''
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    # CREA UN BLOCCO USANDO LA CLASSE DEFINITA SOPRA
    block = Block( 
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    '''
    Inizializza i pesi del modello.
    Vengono previsti 3 casi. I primi due riguardano:
    1) una rete Fully-Connected (per la quale si inizializzano random weights e bias=0)
    2) un modello di Embedding (per il quale i pesi si inizializzano random secondo una normale)
    
    Il terzo caso non ho ben capito.
    
    Comunque sia, questa funzione fa sì che, qualsiasi sia il modelo che si usa, i pesi vengano inizializzati in maniera corretta, in modo da garantire la convergenza del modello.
    '''
    if isinstance(module, nn.Linear):
        # Se il modulo è di tipo nn.Linear
        if module.bias is not None:
            # Se è previsto un bias
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias) # Inizializza i bias a 0
                # Gli altri pesi vengono inizializati automaticamente da nn.Linear
    elif isinstance(module, nn.Embedding):
        # Se invece il modulo è di tipo nn.Embedding
        nn.init.normal_(module.weight, std=initializer_range) # Inizializza pesi random distribuiti secondo una normale con una certa deviazione standard (std)

    if rescale_prenorm_residual: # Se invece c'è questo parametro è True, sa quest'altra tecnica proposta nel " OpenAI GPT-2 Paper Scheme"
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    '''
    Non ho ben capito quando possa essere usata, ma questa funzione inizializza dei pesi, usando una Normale Troncata (nel caso di un nn.Linear), oppure setta i bias=0 e i pesi=1 nel caso di nn.LayerNorm'''
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding.
    Prende in input un'immagine e la proietta in "num_patches" "patch_size"x"patch_size" embeddate, il cui embedding ha dimensione "embed_dim"
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        # La proiezione avviene tramite un layer di 3D Convolution. Come scritto nel paper:
        # "We first use 3D convolution [...] to project the input videos X^v into L non-overlapping spatiotemporal patches X^p "
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )
    
    def forward(self, x):
        x = self.proj(x) # dove proj() è la funzione definita qui sopra
        return x
    

class VisionMamba(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            depth=24, # depth specifica quanti layer (mamba blocks) ci saranno. basta fare: "layer for layer in range(depth)"
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True, # il codice di questo lo trovi in mamba -> modules -> mamba_simple, dove inizia con "forked from https://github.com/hustvl/Vim". Questo parametro lo richiama più giù quando crea diversi Mamba layers
            # video
            kernel_size=1, 
            num_frames=8, 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Si inizializzano ora 3 parametri: cls_token, positional_embedding e temporal_embedding.
        # Ricordiamo infatti che, come scritto nel paper:
        # "The sequence of tokens input to the VideoMamba encoder is:
        # X = [X_cls, X] + pos_embedding + temp_embedding
        # where Xcls is a learnable classification token that is prepended to the start of the sequence."
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) # tensore 1 x 1 x embed_dim pieno di zeri
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim)) # idem ma con dimensioni diverse: POSITIONAL EMBEDDING
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim)) # idem ma con dimensioni diverse
        # NOTE: gli oggetti di tipo nn.Parameter sono sottoclassi di nn.Module. Quindi, in automatico vengono registrati come parametri del modello, e saranno soggetti ad ottimizzazione durante il training
        self.pos_drop = nn.Dropout(p=drop_rate) # p=drop_rate è la probabilità di dropout

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity() # se fc_drop_rate > 0 crea un Dropout Layer. Altrimenti, crea un layer nn.Identity. Come puoi leggere qui: https://stackoverflow.com/questions/64229717/what-is-the-idea-behind-using-nn-identity-for-residual-learning
        # "all nn.Identity does is forwarding the input given to it (basically no-op)"
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity() # qui crea un Linear Layer in cui la dimensione degli input è self.num_features, mentre la dimensione dell'putput è num_classes

        # Create a rule for depth decay: https://paperswithcode.com/method/stochastic-depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # Mamba blocks:
        # come scritto nel paper: "The tokens X are then passed through by L stacked Bi-Mamba blocks"
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs, # il doppio * serve a inserire il contenuto del deizionario come input, senza specificarne uno alla volta
                )
                for i in range(depth)
            ]
        )
        
        # Il paper continua: "The representation of [CLS] token at the final layer is processed by normalization and linear layer for classification."
        # Io credo che il linear layer a cui si riferisce è "head", mentr il NormalizationLayer è quello definito immediatamente qui sotto.
        # Output head: blocco di normalizzazione --> LayerNorm o RMSNorm, con argomenti embed_dim; epsilon_norm; factory_kwargs (device e dtype)
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights) # applica la funzione segm_init_weights a tutti i moduli del modello VisionMamba. Ricorda che questa funzione inizializzava i pesi dei layer, distinguendo tra layer nn.Linear e LayerNorm
        self.head.apply(segm_init_weights) # fa la stessa cosa per head, che dovrebbe essere il layer finale
        trunc_normal_(self.pos_embed, std=.02) # qui inizializza i pesi del layer di embedding

        # mamba init:
        # di preciso non ho capito cosa fa di diverso rispetto alla due funzioni della righe sopra, oltre a star applicando una funzione diversa
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        '''
        Restituisce i nomi di parametri che non devono essere sottoposti a decay durante l'ottimizzazione
        '''
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        '''
        Restituisce il numero di layers del modello
        '''
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        '''
        Qui se ho capito bene carica semplicemente i pesi del modello pre-trainato. Dentro prefix, ci va il path di questo modello
        '''
        _load_weights(self, checkpoint_path, prefix) # questa funzione stava negli import iniziali


    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x) # L'input passa attraverso patch_embed, che proiettava x in embedding delle varie patch
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C) # qui fa solo un reshape per far matchare le dimensioni


        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (stole cls_tokens impl from Phil Wang) qui si crea il class token...
        x = torch.cat((cls_token, x), dim=1) # ... che si concatena ad x...
        x = x + self.pos_embed # ... e infe si sommano i POSITIONAL EMBEDDINGS

        # temporal pos: qui è un grande BHO
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token
        return hidden_states[:, 0, :] # si ritorna solo questo, che quindi dovrebbe rappresentare da solo l'intero input

    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)
        x = self.head(self.head_drop(x))
        return x


def inflate_weight(weight_2d, time_dim, center=True):
    '''
    Prende un tensore 2D di pesi e lo trasforma in 3D
    '''
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


@register_model
def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_small(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=384, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=576, 
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224

    # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
    model = videomamba_middle(num_frames=num_frames).cuda()
    flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda())
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)
