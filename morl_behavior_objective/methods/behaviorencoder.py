import math

import torch as th  # type: ignore[import]
from torch import nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]

# Transformer


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

    def forward(
        self,
        src: th.Tensor,
        src_mask: th.Tensor = None,
        src_key_padding_mask: th.Tensor = None,
        attn_bias: th.Tensor = None,
    ):
        # attn_bias: [H, T, T] or None
        if attn_bias is not None:
            H, T, _ = attn_bias.shape
            B, _L, _ = src.shape
            # expand to [H, B, T, T] then reshape [B*H, T, T]
            bias = attn_bias.unsqueeze(1).expand(H, B, T, T).reshape(H * B, T, T)
            attn_mask = bias
        else:
            attn_mask = src_mask

        attn_output, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, attn_weights


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer_params, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([CustomTransformerEncoderLayer(**encoder_layer_params) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer_params["d_model"])

    def forward(
        self,
        src: th.Tensor,
        src_mask: th.Tensor = None,
        src_key_padding_mask: th.Tensor = None,
        attn_bias: th.Tensor = None,
    ):
        output = src
        attn_weights_list = []
        for layer in self.layers:
            output, attn = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, attn_bias=attn_bias)
            attn_weights_list.append(attn)
        output = self.norm(output)
        return output, attn_weights_list


class FourierFeatureEmbed(nn.Module):
    """Maps coords to high-dim Fourier features."""

    def __init__(self, in_dims=2, num_bands=64, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        # Create fixed bands [1, 2, 4, ..., 2^(num_bands-1)] scaled to max_freq
        bands = 2.0 ** th.linspace(0, math.log2(max_freq), num_bands)
        self.register_buffer("bands", bands)  # [num_bands]

    def forward(self, x):
        x_proj = 2 * math.pi * x.unsqueeze(-1) * self.bands  # broadcast
        x_sin = th.sin(x_proj)  # [B*L, 2, num_bands]
        x_cos = th.cos(x_proj)
        # concat along last dim → [B*L, 2 * num_bands]
        return th.cat([x_sin, x_cos], dim=-1).view(x.shape[0], -1)


class ScaledFourierFeatureEmbed(nn.Module):
    """Deterministic per-dimension Fourier features with learnable per-dim scale.
    Good when you want higher sensitivity without cranking max_freq too high.
    """

    def __init__(self, in_dims=2, num_bands=32, max_freq=32.0, init_log_scale=0.0):
        super().__init__()
        self.in_dims = in_dims
        self.num_bands = num_bands
        bands = 2.0 ** th.linspace(0, math.log2(max_freq), num_bands)
        self.register_buffer("bands", bands)  # [num_bands]
        self.log_scale = nn.Parameter(th.full((in_dims,), init_log_scale))  # learnable

    def forward(self, x: th.Tensor):
        # x: [N, in_dims], assume normalized per-dim
        x_scaled = x * self.log_scale.exp()  # [N, in_dims]
        x_proj = 2 * math.pi * x_scaled.unsqueeze(-1) * self.bands  # [N, in_dims, num_bands]
        return th.cat([th.sin(x_proj), th.cos(x_proj)], dim=-1).reshape(x.shape[0], -1)


class GaussianFourierFeatureEmbed(nn.Module):
    """Random Fourier Features (RFF) with a shared projection across dims:
    z(x) = [sin(2π xW), cos(2π xW)], W ~ N(0, sigma^2).
    Output size is 2*m independent of input dimension; mixes dimensions.
    """

    def __init__(
        self,
        in_dims: int,
        m: int = 512,
        sigma: float = 10.0,
        learnable: bool = False,
        sigmamode: str = "std",
        normalize_out: bool = True,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.m = m
        self.normalize_out = normalize_out
        if sigmamode == "std":
            W = th.randn(in_dims, m) * sigma
        elif sigmamode == "lengthscale":
            W = th.randn(in_dims, m) / (sigma + 1e-8)  # bandwidth via sigma
        self.W = nn.Parameter(W) if learnable else nn.Parameter(W, requires_grad=False)

    def forward(self, x: th.Tensor):
        # x: [N, in_dims], assume normalized
        proj = 2 * math.pi * (x @ self.W)  # [N, m]
        z = th.cat([th.sin(proj), th.cos(proj)], dim=-1)
        if self.normalize_out:
            z = z / math.sqrt(self.m)  # optional variance stabilization
        return z  # [N, 2m]


class CoordMLPEncoder(nn.Module):
    """Embeds 2-D coords into d_model via Fourier features + MLP + LayerNorm,
    with dropout for augmentation.
    Input coords: [B*L, in_dims]
    Output token embeddings: [B*L, d_model].
    """

    def __init__(self, d_model: int, num_bands: int = 64, max_freq: float = 10.0, in_dims: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_dims = in_dims
        # Fourier feature projection (deterministic)
        self.ff = FourierFeatureEmbed(in_dims=self.in_dims, num_bands=num_bands, max_freq=max_freq)
        feat_dim = self.in_dims * 2 * num_bands

        # MLP + LayerNorm + Dropout
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, coords: th.Tensor) -> th.Tensor:
        """coords: [B*L, in_dims]
        returns: [B*L, d_model].
        """
        if coords.shape[-1] != self.in_dims:
            raise ValueError(f"Expected coords with {self.in_dims} dimensions, got {coords.shape[-1]}")
        x = self.ff(coords)  # [B*L, feat_dim]
        return self.net(x)  # [B*L, d_model]


class CoordMLPEncoderScaled(nn.Module):
    """Deterministic bands + learnable per-dimension scale (higher sensitivity without huge max_freq)."""

    def __init__(
        self,
        d_model: int,
        in_dims: int = 2,
        num_bands: int = 32,
        max_freq: float = 32.0,
        init_log_scale: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.ff = ScaledFourierFeatureEmbed(
            in_dims=self.in_dims, num_bands=num_bands, max_freq=max_freq, init_log_scale=init_log_scale
        )
        feat_dim = self.in_dims * 2 * num_bands
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, coords: th.Tensor) -> th.Tensor:
        if coords.shape[-1] != self.in_dims:
            raise ValueError(f"Expected coords with {self.in_dims} dimensions, got {coords.shape[-1]}")
        x = self.ff(coords)
        return self.net(x)


class CoordMLPEncoderGaussian(nn.Module):
    """Random Fourier Features (RFF) using a shared Gaussian projection; mixes dimensions.
    Output feature size is 2*m (independent of in_dims).
    """

    def __init__(
        self,
        d_model: int,
        in_dims: int,
        d_hid: int = 1024,
        m: int = 512,
        sigma: float = 10.0,
        learnable_proj: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.rff = GaussianFourierFeatureEmbed(in_dims=self.in_dims, m=m, sigma=sigma, learnable=learnable_proj)
        feat_dim = 2 * m
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),  # 1. Process in High Dim (128 -> 256)
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim * 2, feat_dim * 2),  # 2. Non-linear mixing (256 -> 256)
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim * 2, d_model),  # 3. Final Squeeze (256 -> 3)
            # nn.LayerNorm(d_model),
        )

    def forward(self, coords: th.Tensor) -> th.Tensor:
        if coords.shape[-1] != self.in_dims:
            raise ValueError(f"Expected coords with {self.in_dims} dimensions, got {coords.shape[-1]}")
        x = self.rff(coords)
        return self.net(x)


# class TemporalConvEncoder(nn.Module):
#     """Takes per-step embeddings [B, L, D] and learns temporal patterns
#     via 1D convolutions over the sequence dimension, with optional dilation.
#     Returns refined embeddings [B, L, D].
#     """

#     def __init__(
#         self,
#         emb_dim: int,
#         hidden_dim: int = 128,
#         kernel_size: int = 3,
#         num_layers: int = 2,
#         use_dilation: bool = True,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         layers = []
#         in_ch = emb_dim
#         for i in range(num_layers):
#             out_ch = hidden_dim if i < num_layers - 1 else emb_dim
#             # compute dilation and padding
#             dilation = 2**i if use_dilation else 1
#             padding = ((kernel_size - 1) // 2) * dilation
#             layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.Dropout(dropout))
#             in_ch = out_ch
#         self.net = nn.Sequential(*layers)

#     def forward(self, x: th.Tensor) -> th.Tensor:
#         # x: [B, L, D] → [B, D, L]
#         x = x.transpose(1, 2)
#         x = self.net(x)  # [B, D, L]
#         x = x.transpose(1, 2)  # [B, L, D]
#         return x


class TemporalConvEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 16,  # Reduced from 2048 to 16 (Safe expansion)
        kernel_size: int = 3,
        num_layers: int = 2,
        use_dilation: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_ch = emb_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else emb_dim
            dilation = 2**i if use_dilation else 1
            padding = ((kernel_size - 1) // 2) * dilation

            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation))

            # CRITICAL FIX: Use GELU instead of ReLU to preserve negative coordinates
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

            in_ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [B, L, D]
        x_in = x.transpose(1, 2)  # [B, D, L]

        # Residual Connection: Output = Input + CNN_Feature
        # This ensures we enrich the token without destroying the original coord
        out = self.net(x_in)

        # If dims match (which they do at the end), add residual
        if out.shape == x_in.shape:
            out = out + x_in

        return out.transpose(1, 2)  # [B, L, D]


class GridEncoderDropout3D(nn.Module):
    """3D CNN encoder over a small temporal window.
    Input: x of shape [B, T, C, H, W]
    Output: per frame embeddings of shape [B, T, out_dim].
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_dim: int = 64,
        hidden_channels: int = 32,  # This is the base number of channels for the first conv
        kernel_size: tuple = (3, 3, 3),  # Default, but you instantiate with (3,3,3)
        pool_kernel: tuple = (1, 2, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        # conv3d: (in_channels, hidden_channels), temporal+spatial
        # Effective padding for kernel (3,3,3) and dilation (2,1,1) should be (2,1,1)
        # If kernel_size is passed as (3,3,3) during instantiation, these padding/dilation values are fine.
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=(2, 1, 1), dilation=(2, 1, 1))
        self.bn1 = nn.BatchNorm3d(hidden_channels)
        # self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=(2,1,1),dilation=(2,1,1))
        # self.bn2   = nn.BatchNorm3d(hidden_channels)
        # self.conv3 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=(2,1,1),dilation=(2,1,1))
        # self.bn3   = nn.BatchNorm3d(hidden_channels)

        # down-sample spatially only
        self.pool = nn.MaxPool3d(pool_kernel)
        self.dropout = nn.Dropout3d(dropout)

        # project to per-frame embedding
        self.adaptpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        # Corrected: Input to fc is hidden_channels*4
        self.fc = nn.Linear(hidden_channels, out_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [B, T, C, H, W] → [B, C, T, H, W]
        x = x.transpose(1, 2)

        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))  # Out channels: hidden_channels
        x = self.pool(x)
        x = self.dropout(x)

        # # # Block 2
        # x = F.relu(self.bn2(self.conv2(x)))       # Out channels: hidden_channels
        # x = self.pool(x)
        # x = self.dropout(x)

        # Block 3
        # x = F.relu(self.bn3(self.conv3(x)))       # Out channels: hidden_channels
        # x = self.pool(x)
        # x = self.dropout(x)

        # collapse spatial dims but keep T
        x = self.adaptpool(x)  # In channels: hidden_channels*4. Out: [B, hidden_channels*4, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B, hidden_channels*4, T]

        # transpose to [B, T, hidden_channels*4]
        x = x.transpose(1, 2)

        # project each frame embedding
        B, T_dim, C_dim = x.shape  # C_dim is hidden_channels*4
        x_flat = x.reshape(B * T_dim, C_dim)
        x_fc_out = self.fc(x_flat)  # fc input is hidden_channels*4
        return x_fc_out.view(B, T_dim, -1)  # [B, T, out_dim]


## Behavior Encoder with SA tokenization and type embeddings


class BehaviorEncoderCLSattnSATyped(nn.Module):
    """A Behavior Encoder using Decision Transformer-style tokenization and positional embeddings.
    It processes sequences of states and actions by interleaving them and assigning a learned
    embedding to each timestep and modality (state/action).
    """

    def __init__(
        self,
        input_channels: int,
        cnn_output_dim: int,
        steps: int,
        nhead: int,
        d_hid: int,
        emb_dim: int,
        num_actions: int,
        nlayers: int = 6,
        dropout: float = 0.1,
        max_len: int = 100,
        input_coord_dims: int = 2,
        max_freq: float = 2.0,
        # NEW: choose coord encoders for states and continuous actions
        coord_state_kind: str = "gaussian",  # {'det','scaled','gaussian'}
        coord_action_kind: str = "gaussian",  # {'det','scaled','gaussian'}
        # NEW: per-kind hyperparams (kept simple; num_bands uses emb_dim by default)
        scaled_init_log_scale_state: float = 0.0,
        scaled_init_log_scale_action: float = 0.0,
        gaussian_m_state: int = 1024,
        gaussian_sigma_state: float = 10.0,
        gaussian_m_action: int = 512,
        gaussian_sigma_action: float = 10.0,
        normalize_output_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = emb_dim
        self.cls_token = nn.Parameter(th.randn(1, 1, emb_dim))
        self.model_type = "BE_SA_DT"
        self.input_coord_dims = input_coord_dims

        # Timestep embedding, as in Decision Transformer. Replaces standard PEs.
        self.embed_timestep = nn.Embedding(max_len, emb_dim)

        # Encoders for states (handles both grid and coordinate-based envs)
        self.cnn_encoder = GridEncoderDropout3D(
            in_channels=input_channels,
            out_dim=cnn_output_dim,
            hidden_channels=32,
            kernel_size=(3, 3, 3),
            pool_kernel=(1, 2, 2),
            dropout=dropout,
        )

        # State coord encoder (selectable)
        self.coord_encoder = self.make_coord_mlp_encoder(
            kind=coord_state_kind,
            d_model=emb_dim,
            d_hid=d_hid,
            in_dims=self.input_coord_dims,
            dropout=dropout,
            num_bands=emb_dim,  # keep your default width
            max_freq=max_freq,  # reuse provided max_freq
            init_log_scale=scaled_init_log_scale_state,
            m=gaussian_m_state,
            sigma=gaussian_sigma_state,
            learnable_proj=True,
        )
        self.temporal_encoder = TemporalConvEncoder(
            emb_dim=emb_dim, hidden_dim=d_hid // 4, kernel_size=7, num_layers=2, dropout=dropout, use_dilation=False
        )

        # Encoder for actions
        self.action_encoder_discr = nn.Embedding(num_actions, emb_dim)

        # Continuous action encoder (selectable)
        self.action_encoder_cont = self.make_coord_mlp_encoder(
            kind=coord_action_kind,
            d_model=emb_dim,
            d_hid=d_hid,
            in_dims=num_actions,
            dropout=dropout,
            num_bands=emb_dim,
            max_freq=max_freq,
            init_log_scale=scaled_init_log_scale_action,
            m=gaussian_m_action,
            sigma=gaussian_sigma_action,
            learnable_proj=True,
        )
        self.action_temporal_encoder_cont = TemporalConvEncoder(
            emb_dim=emb_dim, hidden_dim=d_hid // 4, kernel_size=7, num_layers=2, dropout=dropout, use_dilation=False
        )
        self.action_temporal_encoder_discr = TemporalConvEncoder(
            emb_dim=emb_dim, hidden_dim=d_hid // 4, kernel_size=7, num_layers=2, dropout=dropout, use_dilation=False
        )

        # Modality embeddings to differentiate states and actions
        self.state_type_embedding = nn.Parameter(th.randn(1, 1, emb_dim))
        self.action_type_embedding = nn.Parameter(th.randn(1, 1, emb_dim))

        self.input_proj = nn.Linear(cnn_output_dim, emb_dim) if cnn_output_dim != emb_dim else nn.Identity()

        encoder_layer_params = {
            "d_model": self.d_model,
            "nhead": nhead,
            "dim_feedforward": d_hid,
            "dropout": dropout,
            "activation": "gelu",
            "batch_first": True,
        }
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer_params, nlayers)
        self._dropout_p = dropout
        self._register_dropout_modules()
        # self.pooling = MHAPooling(emb_dim, num_heads=nhead)
        self.init_weights()
        print(
            f"Input treated with {coord_state_kind} fourier feature encoder for states and {coord_action_kind} fourier feature encoder for actions."
        )
        self.normalize_output_embeddings = normalize_output_embeddings

    @staticmethod
    def make_coord_mlp_encoder(
        kind: str,
        d_model: int,
        d_hid: int,
        in_dims: int,
        dropout: float = 0.1,
        # deterministic/scaled
        num_bands: int = 64,
        max_freq: float = 10.0,
        init_log_scale: float = 0.0,
        # gaussian
        m: int = 512,
        sigma: float = 10.0,
        learnable_proj: bool = False,
    ) -> nn.Module:
        r"""Kind \in {'det','scaled','gaussian'}."""
        kind = kind.lower()
        if kind == "det":
            return CoordMLPEncoder(d_model, in_dims=in_dims, num_bands=num_bands, max_freq=max_freq, dropout=dropout)
        if kind == "scaled":
            return CoordMLPEncoderScaled(
                d_model,
                in_dims=in_dims,
                num_bands=num_bands,
                max_freq=max_freq,
                init_log_scale=init_log_scale,
                dropout=dropout,
            )
        if kind == "gaussian":
            return CoordMLPEncoderGaussian(
                d_model, in_dims=in_dims, d_hid=d_hid, m=m, sigma=sigma, learnable_proj=learnable_proj, dropout=dropout
            )
        raise ValueError(f"Unknown coord encoder kind: {kind}")

    def _register_dropout_modules(self):
        self._dropouts = []
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                self._dropouts.append(m)

    def set_dropout(self, p: float):
        for dr in self._dropouts:
            dr.p = p
        self._dropout_p = p

    # def init_weights(self) -> None:
    #     for name, p in self.named_parameters():
    #         # Skip the Fourier (RFF) weights!
    #         if "rff.W" in name or "ff.log_scale" in name:
    #             continue

    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def init_weights(self) -> None:
        """Initialize weights properly, respecting special layers."""
        for _name, module in self.named_modules():
            # Skip RFF layers - they have special initialization
            if isinstance(module, GaussianFourierFeatureEmbed):
                continue  # W is already initialized as N(0, σ²)

            # Skip normalization layers
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue  # Default init (weight=1, bias=0) is correct

            # Skip embeddings
            if isinstance(module, nn.Embedding):
                continue  # Default init is fine, or use:
                # nn.init.normal_(module.weight, mean=0, std=0.02)

            # Initialize Linear layers
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # Initialize Conv layers
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Special: CLS token and type embeddings
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        nn.init.normal_(self.state_type_embedding, mean=0, std=0.02)
        nn.init.normal_(self.action_type_embedding, mean=0, std=0.02)

    def forward(self, states: th.Tensor, actions: th.Tensor, src_key_padding_mask: th.Tensor | None = None) -> tuple:
        B, T, *_ = states.shape

        # 1. Encode states based on their dimension (grid vs. coord)
        if states.dim() == 5:
            state_emb = self.cnn_encoder(states.view(B * T, *states.shape[2:])).view(B, T, -1)
        elif states.dim() == 3:
            state_emb = self.coord_encoder(states.view(B * T, -1)).view(B, T, -1)
            state_emb = self.temporal_encoder(state_emb)
        else:
            raise ValueError(f"Unsupported state dimension: {states.dim()}")
        state_emb = self.input_proj(state_emb)

        if actions[0][0].shape[0] == 1:
            # Discrete actions
            action_emb = self.action_encoder_discr(actions.long())
            action_emb = action_emb.view(B, T, self.d_model)
            action_emb = self.action_temporal_encoder_cont(action_emb)
        else:
            # Continuous actions
            flat_act = actions.view(B * T, -1)
            action_emb = self.action_encoder_cont(flat_act).view(B, T, self.d_model)
            action_emb = self.action_temporal_encoder_cont(action_emb)

        # 3. Add timestep and modality embeddings
        timesteps = th.arange(T, device=states.device).expand(B, T)
        timestep_embeddings = self.embed_timestep(timesteps)

        state_emb = state_emb + timestep_embeddings + self.state_type_embedding
        action_emb = action_emb + timestep_embeddings + self.action_type_embedding

        # 4. Interleave state and action embeddings to form the input sequence
        # -> [s0, a0, s1, a1, ...]
        interleaved_emb = th.stack([state_emb, action_emb], dim=2).view(B, 2 * T, self.d_model)

        # 5. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        emb = th.cat([cls, interleaved_emb], dim=1)

        # 6. Adjust padding mask for the new interleaved sequence
        final_padding_mask = None
        if src_key_padding_mask is not None:
            cls_mask = th.zeros(B, 1, dtype=th.bool, device=src_key_padding_mask.device)
            interleaved_mask = src_key_padding_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(B, 2 * T)
            final_padding_mask = th.cat([cls_mask, interleaved_mask], dim=1)

        # 7. Transformer pass

        transformer_out, attn_list = self.transformer_encoder(emb, src_mask=None, src_key_padding_mask=final_padding_mask)

        # 8. Normalize outputs and get trajectory summary
        if self.normalize_output_embeddings:
            norm = transformer_out.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            normalized = transformer_out / norm
        else:
            normalized = transformer_out

        # pooled, pool_weights = self.pooling(transformer_out, final_padding_mask)
        cls_emb = normalized[:, 0, :]

        stacked_attns = th.stack(attn_list)
        cls_attn = stacked_attns[:, :, :, 0, :].mean(dim=(0, 2))
        attn_list_agg = stacked_attns.sum(dim=0).sum(dim=1)

        return normalized, attn_list_agg, _, _, cls_emb, cls_attn


class BehaviorEncoderMLPDouble(nn.Module):
    """A simple MLP baseline that flattens trajectories and processes them directly.
    No RFF, no CNN, no temporal encoding, no transformer - just MLPs.
    """

    def __init__(
        self,
        input_channels: int,  # Kept for API compatibility
        cnn_output_dim: int,  # Kept for API compatibility
        steps: int,  # Max trajectory length
        nhead: int,  # Kept for API compatibility, not used
        d_hid: int,
        emb_dim: int,
        num_actions: int,
        nlayers: int = 6,  # Kept for API compatibility
        dropout: float = 0.1,
        max_len: int = 100,
        input_coord_dims: int = 2,
        # All other kwargs kept for API compatibility but ignored
        max_freq: float = 2.0,
        coord_state_kind: str = "gaussian",
        coord_action_kind: str = "gaussian",
        scaled_init_log_scale_state: float = 0.0,
        scaled_init_log_scale_action: float = 0.0,
        gaussian_m_state: int = 1024,
        gaussian_sigma_state: float = 10.0,
        gaussian_m_action: int = 512,
        gaussian_sigma_action: float = 10.0,
        normalize_output_embeddings: bool = False,
    ):
        super().__init__()
        self.d_model = emb_dim
        self.model_type = "BE_MLP_Double"
        self.input_coord_dims = input_coord_dims
        self.max_len = max_len
        self.num_actions = num_actions
        self.normalize_output_embeddings = normalize_output_embeddings

        # Calculate input dimension for flattened trajectory
        # Each timestep has: state (input_coord_dims) + action (1 for discrete, num_actions for continuous)
        # We'll handle both cases in forward()
        self.state_dim = input_coord_dims

        # For discrete actions: one-hot encode to num_actions dims
        # For continuous actions: use num_actions dims directly
        self.action_dim = num_actions

        # Flattened input size per timestep
        self.per_step_dim = self.state_dim + self.action_dim
        self.flat_input_dim = max_len * self.per_step_dim

        # For returning local tokens (needed for DIM loss compatibility)
        # We'll create pseudo-tokens by chunking the hidden representation
        self.per_token_encoder = nn.Sequential(
            nn.Linear(self.per_step_dim, d_hid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid // 2, emb_dim),
        )

        # Aggregation from per-token representations
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(emb_dim, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, emb_dim),
        )

        self._dropout_p = dropout
        self._register_dropout_modules()
        self.init_weights()

        print(f"[MLP Baseline] Simple flattened MLP encoder. Input dim: {self.flat_input_dim}, Output dim: {emb_dim}")

    def _register_dropout_modules(self):
        self._dropouts = []
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                self._dropouts.append(m)

    def set_dropout(self, p: float):
        for dr in self._dropouts:
            dr.p = p
        self._dropout_p = p

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, states: th.Tensor, actions: th.Tensor, src_key_padding_mask: th.Tensor | None = None) -> tuple:
        B, T, *_ = states.shape
        device = states.device

        # 1. Flatten states
        if states.dim() == 5:
            states_flat = states.view(B, T, -1)
            if states_flat.shape[-1] > self.state_dim:
                states_flat = states_flat[..., : self.state_dim]
        elif states.dim() == 3:
            states_flat = states
        else:
            raise ValueError(f"Unsupported state dimension: {states.dim()}")

        # 2. Process actions
        if actions.dim() == 2:
            actions = actions.unsqueeze(-1)

        is_discrete = actions.shape[-1] == 1

        if is_discrete:
            actions_long = actions.long().squeeze(-1)
            actions_flat = F.one_hot(actions_long, num_classes=self.num_actions).float()
        else:
            actions_flat = actions

        # 3. Concatenate per timestep [B, T, state_dim + action_dim]
        sa_per_step = th.cat([states_flat, actions_flat], dim=-1)

        # 4. Encode EACH token through shared MLP -> [B, T, emb_dim]
        # This creates tokens that are used for both DIM loss AND aggregation
        token_embeddings = self.per_token_encoder(sa_per_step)  # [B, T, emb_dim]

        # 5. Apply mask for aggregation
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).float().unsqueeze(-1)  # [B, T, 1]
            masked_tokens = token_embeddings * valid_mask
            pooled = masked_tokens.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = token_embeddings.mean(dim=1)

        # 6. Aggregate to get CLS-like embedding
        cls_emb = self.aggregation_mlp(pooled)

        # 7. Normalize
        if self.normalize_output_embeddings:
            cls_emb = F.normalize(cls_emb, p=2, dim=-1)
            token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)

        # 8. Interleave tokens for DIM loss compatibility [B, 2*T, emb_dim]
        # Split token_embeddings conceptually into state and action parts
        # Since we encoded (s,a) pairs together, we duplicate for interleaving
        interleaved_tokens = token_embeddings.unsqueeze(2).expand(-1, -1, 2, -1).reshape(B, 2 * T, self.d_model)

        dummy_attn = th.zeros(B, 2 * T, device=device)

        return interleaved_tokens, dummy_attn, None, None, cls_emb, dummy_attn


class BehaviorEncoderMLPBaseline(nn.Module):
    """Simplest MLP baseline that supports DIM loss.

    Architecture:
        (s,a) pairs → shared MLP → tokens → mean pool → CLS

    Only ONE network (token_net). CLS = mean(tokens).
    No separate aggregation network.
    """

    def __init__(
        self,
        input_channels: int,
        cnn_output_dim: int,
        steps: int,
        nhead: int,
        d_hid: int,
        emb_dim: int,
        num_actions: int,
        nlayers: int = 6,
        dropout: float = 0.1,
        max_len: int = 100,
        normalize_output_embeddings: bool = True,
        input_coord_dims: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.d_model = emb_dim
        self.model_type = "BE_MLP_Simple"
        self.input_coord_dims = input_coord_dims
        self.max_len = max_len
        self.num_actions = num_actions
        self.normalize_output_embeddings = normalize_output_embeddings

        self.state_dim = input_coord_dims
        self.action_dim = num_actions
        self.per_step_dim = self.state_dim + self.action_dim

        # Single network: (s,a) → token embedding
        self.token_net = nn.Sequential(
            nn.Linear(self.per_step_dim, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, emb_dim),
        )

        self._dropout_p = dropout
        self._register_dropout_modules()
        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MLP Simple] (s,a) → MLP → tokens → mean → CLS | Params: {n_params:,}")

    def _register_dropout_modules(self):
        self._dropouts = []
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                self._dropouts.append(m)

    def set_dropout(self, p: float):
        for dr in self._dropouts:
            dr.p = p
        self._dropout_p = p

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, states: th.Tensor, actions: th.Tensor, src_key_padding_mask: th.Tensor | None = None) -> tuple:
        B, T, *_ = states.shape
        device = states.device

        # 1. Flatten states
        if states.dim() == 5:
            states_flat = states.view(B, T, -1)[..., : self.state_dim]
        elif states.dim() == 3:
            states_flat = states
        else:
            raise ValueError(f"Unsupported state dimension: {states.dim()}")

        # 2. Process actions
        if actions.dim() == 2:
            actions = actions.unsqueeze(-1)

        if actions.shape[-1] == 1:  # Discrete
            actions_flat = F.one_hot(actions.long().squeeze(-1), num_classes=self.num_actions).float()
        else:  # Continuous
            actions_flat = actions

        # 3. Concatenate per timestep [B, T, per_step_dim]
        sa = th.cat([states_flat, actions_flat], dim=-1)

        # 4. Apply single token network to each timestep
        # [B, T, per_step_dim] → [B, T, emb_dim]
        token_embeddings = self.token_net(sa)

        # 5. CLS = masked mean of tokens
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).float().unsqueeze(-1)  # [B, T, 1]
            masked_tokens = token_embeddings * valid_mask
            cls_emb = masked_tokens.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            cls_emb = token_embeddings.mean(dim=1)  # [B, emb_dim]

        # 6. Normalize if required
        if self.normalize_output_embeddings:
            cls_emb = F.normalize(cls_emb, p=2, dim=-1)
            token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)

        # 7. Interleave tokens for DIM loss compatibility [B, 2*T, emb_dim]
        interleaved_tokens = token_embeddings.unsqueeze(2).expand(-1, -1, 2, -1).reshape(B, 2 * T, self.d_model)

        dummy_attn = th.zeros(B, 2 * T, device=device)

        return interleaved_tokens, dummy_attn, None, None, cls_emb, dummy_attn


class BehaviorEncoderMLPWithLocalTokens(nn.Module):
    """MLP baseline that processes the FULL trajectory at once (like Transformer)
    but creates local tokens for DIM loss compatibility.

    Architecture:
        Full trajectory → Flatten → MLP → Split into tokens
        CLS = mean(tokens)
    """

    def __init__(
        self,
        input_channels: int,
        cnn_output_dim: int,
        steps: int,
        nhead: int,
        d_hid: int,
        emb_dim: int,
        num_actions: int,
        nlayers: int = 6,
        dropout: float = 0.1,
        max_len: int = 100,
        input_coord_dims: int = 2,
        normalize_output_embeddings: bool = True,
        num_local_tokens: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.d_model = emb_dim
        self.model_type = "BE_MLP_LocalTokens"
        self.input_coord_dims = input_coord_dims
        self.max_len = max_len
        self.num_actions = num_actions
        self.normalize_output_embeddings = normalize_output_embeddings
        self.num_local_tokens = num_local_tokens

        self.state_dim = input_coord_dims
        self.action_dim = num_actions
        self.per_step_dim = self.state_dim + self.action_dim
        self.flat_input_dim = max_len * self.per_step_dim

        # --- Main Network: Full trajectory → hidden → tokens ---
        # We output directly to (num_local_tokens * emb_dim) so we can split
        hidden_dim = d_hid * 2

        self.encoder_net = nn.Sequential(
            nn.Linear(self.flat_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_local_tokens * emb_dim),  # Direct output
        )

        self._dropout_p = dropout
        self._register_dropout_modules()
        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[MLP LocalTokens] Full traj → {num_local_tokens} tokens (dim={emb_dim}) | CLS=mean(tokens) | Params: {n_params:,}"
        )

    def _register_dropout_modules(self):
        self._dropouts = []
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                self._dropouts.append(m)

    def set_dropout(self, p: float):
        for dr in self._dropouts:
            dr.p = p
        self._dropout_p = p

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, states: th.Tensor, actions: th.Tensor, src_key_padding_mask: th.Tensor | None = None) -> tuple:
        B, T, *_ = states.shape
        device = states.device

        # 1. Flatten states
        if states.dim() == 5:
            states_flat = states.view(B, T, -1)[..., : self.state_dim]
        elif states.dim() == 3:
            states_flat = states
        else:
            raise ValueError(f"Unsupported state dimension: {states.dim()}")

        # 2. Process actions
        if actions.dim() == 2:
            actions = actions.unsqueeze(-1)

        if actions.shape[-1] == 1:  # Discrete
            actions_flat = F.one_hot(actions.long().squeeze(-1), num_classes=self.num_actions).float()
        else:  # Continuous
            actions_flat = actions

        # 3. Concatenate and flatten ENTIRE trajectory
        sa = th.cat([states_flat, actions_flat], dim=-1)

        # Apply mask by zeroing out padded steps
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).float().unsqueeze(-1)
            sa = sa * valid_mask

        flat_traj = sa.reshape(B, -1)  # [B, flat_input_dim]

        # 4. Process full trajectory → tokens
        # [B, flat_input_dim] → [B, num_local_tokens * emb_dim]
        hidden = self.encoder_net(flat_traj)

        # 5. Reshape to get local tokens [B, num_local_tokens, emb_dim]
        local_tokens = hidden.view(B, self.num_local_tokens, self.d_model)

        # 6. CLS = mean of all tokens
        cls_emb = local_tokens.mean(dim=1)  # [B, emb_dim]

        # 7. Normalize if required
        if self.normalize_output_embeddings:
            cls_emb = F.normalize(cls_emb, p=2, dim=-1)
            local_tokens = F.normalize(local_tokens, p=2, dim=-1)

        # 8. Interleave tokens for DIM loss compatibility [B, 2*num_local_tokens, emb_dim]
        interleaved_tokens = (
            local_tokens.unsqueeze(2).expand(-1, -1, 2, -1).reshape(B, 2 * self.num_local_tokens, self.d_model)
        )

        dummy_attn = th.zeros(B, 2 * self.num_local_tokens, device=device)

        return interleaved_tokens, dummy_attn, None, None, cls_emb, dummy_attn


class BasicMLPEncoder(nn.Module):
    """A Global MLP baseline that flattens the ENTIRE trajectory into one vector
    and maps it directly to a single embedding (CLS).

    Architecture:
        [s0, a0, s1, a1, ..., sT, aT] (Flattened)
                    ↓
              MLP (Linear -> GELU -> ...)
                    ↓
              Single Embedding [B, emb_dim]

    Note: This model does NOT support DIM loss meaningfully because it does not
    produce local tokens, only a global summary.
    """

    def __init__(
        self,
        input_channels: int,
        cnn_output_dim: int,
        steps: int,
        nhead: int,
        d_hid: int,
        emb_dim: int,
        num_actions: int,
        nlayers: int = 6,
        dropout: float = 0.1,
        max_len: int = 100,
        input_coord_dims: int = 2,
        normalize_output_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = emb_dim
        self.model_type = "BE_MLP_Global"
        self.input_coord_dims = input_coord_dims
        self.max_len = max_len
        self.num_actions = num_actions
        self.normalize_output_embeddings = normalize_output_embeddings

        self.state_dim = input_coord_dims
        self.action_dim = num_actions

        # Calculate input dimension: T * (state + action)
        self.flat_input_dim = max_len * (self.state_dim + self.action_dim)

        # The Global Encoder MLP
        self.encoder_net = nn.Sequential(
            nn.Linear(self.flat_input_dim, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, emb_dim),
        )

        self._dropout_p = dropout
        self._register_dropout_modules()
        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MLP Global] Full traj (dim={self.flat_input_dim}) → MLP → CLS | Params: {n_params:,}")

    def _register_dropout_modules(self):
        self._dropouts = []
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                self._dropouts.append(m)

    def set_dropout(self, p: float):
        for dr in self._dropouts:
            dr.p = p
        self._dropout_p = p

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, states: th.Tensor, actions: th.Tensor, src_key_padding_mask: th.Tensor | None = None) -> tuple:
        B, T, *_ = states.shape
        device = states.device

        # 1. Flatten states
        if states.dim() == 5:
            states_flat = states.view(B, T, -1)[..., : self.state_dim]
        elif states.dim() == 3:
            states_flat = states
        else:
            raise ValueError(f"Unsupported state dimension: {states.dim()}")

        # 2. Process actions
        if actions.dim() == 2:
            actions = actions.unsqueeze(-1)

        if actions.shape[-1] == 1:  # Discrete
            actions_flat = F.one_hot(actions.long().squeeze(-1), num_classes=self.num_actions).float()
        else:  # Continuous
            actions_flat = actions

        # 3. Concatenate [B, T, state_dim + action_dim]
        sa = th.cat([states_flat, actions_flat], dim=-1)

        # 4. Apply Masking BEFORE flattening
        # This is crucial: we must zero out padded steps so they don't affect the MLP
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).float().unsqueeze(-1)  # [B, T, 1]
            sa = sa * valid_mask

        # 5. Flatten entire trajectory [B, T * (state+action)]
        flat_traj = sa.reshape(B, -1)

        cur_dim = flat_traj.shape[1]
        if cur_dim < self.flat_input_dim:
            pad = th.zeros(B, self.flat_input_dim - cur_dim, dtype=flat_traj.dtype, device=device)
            flat_traj = th.cat([flat_traj, pad], dim=1)
        elif cur_dim > self.flat_input_dim:
            flat_traj = flat_traj[:, : self.flat_input_dim]

        # 6. Pass through MLP to get CLS embedding
        cls_emb = self.encoder_net(flat_traj)  # [B, emb_dim]

        # 7. Normalize if required
        if self.normalize_output_embeddings:
            cls_emb = F.normalize(cls_emb, p=2, dim=-1)

        # 8. API Compatibility Return
        # We return dummy tokens because this model doesn't have local tokens.
        # The shape [B, 2*T, emb_dim] ensures the training loop doesn't crash if it checks shapes,
        # but DIM loss should be disabled (weight=0) when using this model.
        dummy_tokens = th.zeros(B, 2 * T, self.d_model, device=device)
        dummy_attn = th.zeros(B, 2 * T, device=device)

        return dummy_tokens, dummy_attn, None, None, cls_emb, dummy_attn

class BehaviorEncoderLSTMBaseline(nn.Module):
    """
    LSTM-based baseline encoder.

    Architecture:
        (s_t, a_t) -> per-step MLP -> LSTM(d_hid) -> CLS projection -> emb_dim

    This keeps the external embedding dimension small (e.g. 3), but gives the
    recurrent backbone enough capacity to be comparable to the MLP baseline.
    """

    def __init__(
        self,
        input_channels: int,
        cnn_output_dim: int,
        steps: int,
        nhead: int,
        d_hid: int,
        emb_dim: int,
        num_actions: int,
        nlayers: int = 2,
        dropout: float = 0.1,
        max_len: int = 100,
        input_coord_dims: int = 2,
        normalize_output_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = emb_dim
        self.model_type = "BE_LSTM"
        self.input_coord_dims = input_coord_dims
        self.max_len = max_len
        self.num_actions = num_actions
        self.normalize_output_embeddings = normalize_output_embeddings

        self.state_dim = input_coord_dims
        self.action_dim = num_actions
        self.per_step_dim = self.state_dim + self.action_dim

        # Per-step feature extractor before recurrent modeling.
        self.step_encoder = nn.Sequential(
            nn.Linear(self.per_step_dim, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Recurrent backbone with hidden size d_hid, not emb_dim.
        self.lstm = nn.LSTM(
            input_size=d_hid,
            hidden_size=d_hid,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout if nlayers > 1 else 0.0,
            bidirectional=False,
        )

        # Project recurrent features down to the requested embedding size.
        self.token_proj = nn.Linear(d_hid, emb_dim)
        self.cls_proj = nn.Linear(d_hid, emb_dim)

        self._dropout_p = dropout
        self._register_dropout_modules()
        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LSTM Baseline] (s,a) -> MLP -> LSTM(d_hid) -> CLS | Params: {n_params:,}")

    def _register_dropout_modules(self):
        self._dropouts = []
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                self._dropouts.append(module)

    def set_dropout(self, p: float):
        for dropout_layer in self._dropouts:
            dropout_layer.p = p
        self._dropout_p = p

    def init_weights(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1 and "weight_hh" not in name:
                nn.init.xavier_uniform_(parameter)
        for name, parameter in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(parameter)

    def _prepare_states(self, states: th.Tensor) -> th.Tensor:
        B, T, *_ = states.shape
        if states.dim() == 5:
            return states.view(B, T, -1)[..., : self.state_dim]
        if states.dim() == 3:
            return states
        raise ValueError(f"Unsupported state dimension: {states.dim()}")

    def _prepare_actions(self, actions: th.Tensor) -> th.Tensor:
        if actions.dim() == 2:
            actions = actions.unsqueeze(-1)

        if actions.shape[-1] == 1:
            return F.one_hot(actions.long().squeeze(-1), num_classes=self.num_actions).float()

        return actions

    def forward(self, states: th.Tensor, actions: th.Tensor, src_key_padding_mask: th.Tensor | None = None) -> tuple:
        B, T, *_ = states.shape
        device = states.device

        states_flat = self._prepare_states(states)
        actions_flat = self._prepare_actions(actions)
        sa = th.cat([states_flat, actions_flat], dim=-1)  # [B, T, per_step_dim]

        step_feats = self.step_encoder(sa)  # [B, T, d_hid]

        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=1).to(th.int64).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                step_feats, lengths=lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, (h_n, _) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            lstm_out, (h_n, _) = self.lstm(step_feats)

        # Final hidden state from the last LSTM layer.
        cls_hidden = h_n[-1]  # [B, d_hid]
        cls_emb = self.cls_proj(cls_hidden)  # [B, emb_dim]

        token_embeddings = self.token_proj(lstm_out)  # [B, T, emb_dim]

        if self.normalize_output_embeddings:
            cls_emb = F.normalize(cls_emb, p=2, dim=-1)
            token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)

        # DIM-compatible token shape: [B, 2*T, emb_dim]
        interleaved_tokens = (
            token_embeddings.unsqueeze(2)
            .expand(-1, -1, 2, -1)
            .reshape(B, 2 * T, self.d_model)
        )

        dummy_attn = th.zeros(B, 2 * T, device=device)

        return interleaved_tokens, dummy_attn, None, None, cls_emb, dummy_attn

# Decoder
class TrajectoryDecoder(nn.Module):
    def __init__(self, emb_dim, state_dim, action_dim, max_len):  # Removed spec_norm
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.out_features = max_len * (state_dim + action_dim)

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 1024),  # Removed spec_norm logic
            nn.GELU(),
            nn.Linear(1024, 1024),  # Removed spec_norm logic
            nn.GELU(),
            nn.Linear(1024, self.out_features),  # Removed spec_norm logic
        )

    def forward(self, cls_embedding):
        # cls_embedding shape: [B, emb_dim]
        flat_recon = self.net(cls_embedding)
        # flat_recon shape: [B, max_len * (state_dim + action_dim)]

        # Reshape to [B, max_len, state_dim + action_dim]
        recon = flat_recon.view(-1, self.max_len, self.state_dim + self.action_dim)

        states = recon[..., : self.state_dim]
        actions = recon[..., self.state_dim :]
        return states, actions


# Policy-Level Set Encoder
class PolicySetEncoder(nn.Module):
    """A permutation-invariant encoder for a *set* of trajectory embeddings.
    Takes a set of [N, D] embeddings (where D is usually small, e.g. 3),
    projects them to high-dim for processing, and outputs a single [1, D] embedding.
    """

    def __init__(
        self,
        emb_dim: int,  # Input/Output dimension (e.g., 3)
        hidden_dim: int,  # FFN dimension (e.g., 1024)
        internal_dim: int = 3,  # NEW: Transformer working dimension
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.internal_dim = internal_dim

        # 1. Input Projection: Expand 3D input -> 64D working space
        self.input_proj = nn.Linear(emb_dim, internal_dim)

        # 2. Learnable [CLS] token in High-Dim space
        self.cls_token = nn.Parameter(th.randn(1, 1, internal_dim))

        # 3. Transformer Encoder (High Capacity)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=internal_dim,  # Now 64
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Output Projection: Squash 64D summary -> 3D Output
        self.output_proj = nn.Linear(internal_dim, emb_dim)

    def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Input: x (th.Tensor): Set of trajectory embeddings. Shape: [N_trajs, emb_dim]."""
        # Add batch dimension -> [1, N_trajs, emb_dim]
        x = x.unsqueeze(0)

        # 1. Project inputs up to internal dimension
        # [1, N_trajs, emb_dim] -> [1, N_trajs, internal_dim]
        x_high = self.input_proj(x)

        # 2. Prepend High-Dim CLS token
        cls_tokens = self.cls_token.expand(x_high.shape[0], -1, -1)
        x_high = th.cat((cls_tokens, x_high), dim=1)  # [1, 1+N_trajs, internal_dim]

        # 3. Pass through transformer
        all_tokens_high = self.transformer_encoder(x_high)  # [1, 1+N_trajs, internal_dim]

        # 4. Project BACK to low dimension for output
        all_tokens_low = self.output_proj(all_tokens_high)  # [1, 1+N_trajs, emb_dim]

        # Get the CLS token output (the learned policy representation)
        cls_output = all_tokens_low[:, 0, :]  # [1, emb_dim]

        # Optional: Normalize if you want the policy embedding to stay on the sphere
        cls_output = F.normalize(cls_output, p=2, dim=1)

        return cls_output, all_tokens_low


# DeepInfoMax Loss and InfoNCE Loss
class Discriminator(nn.Module):
    """A simple MLP to distinguish between positive and negative pairs."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Linear(input_dim // 2, 1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class DeepInfoMaxLoss(nn.Module):
    """Deep InfoMax loss using a discriminator.
    Maximizes MI between a global summary vector and its local feature vectors.
    This version uses the Jensen-Shannon Divergence (JSD) estimator.
    """

    def __init__(self, encoder_dim: int):
        super().__init__()
        # The discriminator takes a concatenated global and local vector
        self.discriminator = Discriminator(input_dim=encoder_dim * 2)
        self.softplus = nn.Softplus()

    def forward(self, global_emb: th.Tensor, local_embs: th.Tensor, key_padding_mask: th.Tensor):
        """Args:
        global_emb (Tensor): The [CLS] embedding. Shape: [B, D_emb]
        local_embs (Tensor): The full sequence of token embeddings. Shape: [B, T, D_emb]
        key_padding_mask (Tensor): Mask for padded tokens. Shape: [B, T], True if padded.
        """
        _B, T, _D = local_embs.shape
        # Ensure the key_padding_mask matches the sequence length of local_embs
        if key_padding_mask.size(1) != local_embs.size(1):
            raise ValueError("Mismatch between mask and interleaved sequence length.")

        # --- Positive Pairs (Global summary with its own local features) ---
        global_emb_expanded = global_emb.unsqueeze(1).expand(-1, T, -1)
        positive_pairs = th.cat((global_emb_expanded, local_embs), dim=-1)  # [B, T, 2*D]
        positive_scores = self.discriminator(positive_pairs).squeeze(-1)  # [B, T]

        # --- Negative Pairs (Global summary with shuffled local features) ---
        # Shuffle local embeddings across the batch dimension to create negative samples
        shuffled_local_embs = th.cat((local_embs[1:], local_embs[0:1]), dim=0)
        negative_pairs = th.cat((global_emb_expanded, shuffled_local_embs), dim=-1)
        negative_scores = self.discriminator(negative_pairs).squeeze(-1)  # [B, T]

        # --- Calculate JSD-based Loss ---
        # Loss for positive pairs: E[softplus(-D(positive_pairs))]
        # We want the discriminator score to be high, so -score should be low, and softplus(-score) should be low.
        loss_pos = self.softplus(-positive_scores)

        # Loss for negative pairs: E[softplus(D(negative_pairs))]
        # We want the discriminator score to be low, so softplus(score) should be low.
        loss_neg = self.softplus(negative_scores)

        # Combine and mask the loss
        # We only compute the loss over non-padded tokens.
        loss_unmasked = loss_pos + loss_neg

        # Create a mask for valid (non-padded) tokens
        valid_token_mask = ~key_padding_mask

        # Apply the mask and compute the mean loss over valid tokens
        return loss_unmasked[valid_token_mask].mean()


class InstanceLoss(nn.Module):
    def __init__(self, temperature, device):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.loss_type = "IL"

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, current_batch_size):
        N = 2 * current_batch_size
        mask = th.ones((N, N), dtype=th.bool, device=self.device)
        mask = mask.fill_diagonal_(False)  # Remove self-comparisons
        for i in range(current_batch_size):
            # Remove positive pairs (z_i[k] vs z_j[k] and z_j[k] vs z_i[k])
            mask[i, current_batch_size + i] = False
            mask[current_batch_size + i, i] = False
        return mask

    def forward(self, z_i, z_j):
        current_batch_size = z_i.shape[0]

        if current_batch_size == 0:
            # Or handle as appropriate, e.g., if z_j.shape[0] is also 0
            return th.tensor(0.0, device=self.device, requires_grad=True)

        N = 2 * current_batch_size
        z = th.cat((z_i, z_j), dim=0)

        sim = th.matmul(z, z.T) / self.temperature
        sim_i_j = th.diag(sim, current_batch_size)
        sim_j_i = th.diag(sim, -current_batch_size)

        positive_samples = th.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_mask = self.mask_correlated_samples(current_batch_size)
        negative_samples = sim[negative_mask].reshape(N, -1)

        labels = th.zeros(N).to(positive_samples.device).long()
        logits = th.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class VarianceCovarianceLoss(nn.Module):
    """Forces embeddings to span the full D-dimensional space (Variance)
    and ensures dimensions are orthogonal (Covariance).
    Based on VICReg.
    """

    def __init__(self, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z):
        # z: [Batch, Dim]
        batch_size, num_features = z.shape

        # 1. Centering
        z = z - z.mean(dim=0)

        # 2. Variance Loss: Force std of each dim to be close to 1.0
        # This prevents collapse (all points mapping to 0 or a single line)
        std_z = th.sqrt(z.var(dim=0) + 0.0001)
        std_loss = th.mean(F.relu(1 - std_z))

        # 3. Covariance Loss: Force off-diagonal covariances to 0
        # This prevents all 3 dimensions from being correlated (forming a line)
        cov_z = (z.T @ z) / (batch_size - 1)
        off_diag_mask = ~th.eye(num_features, device=z.device, dtype=th.bool)
        cov_loss = cov_z[off_diag_mask].pow(2).sum() / num_features

        return self.std_coeff * std_loss + self.cov_coeff * cov_loss


class VICRegLoss(nn.Module):
    """Full VICReg loss with all three components:
    - Invariance: MSE between two views of the same sample (requires augmentation)
    - Variance: Force each dimension to have std >= 1
    - Covariance: Force dimensions to be uncorrelated.

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization"
    """

    def __init__(
        self,
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps = eps

    def forward(self, z1: th.Tensor, z2: th.Tensor) -> tuple[th.Tensor, dict]:
        """Compute full VICReg loss between two views.

        Args:
            z1: First view embeddings [B, D] (NOT normalized)
            z2: Second view embeddings [B, D] (NOT normalized)

        Returns:
            total_loss: Weighted sum of all components
            loss_dict: Dictionary with individual loss values for logging
        """
        B, D = z1.shape

        # === 1. Invariance Loss ===
        # MSE between the two views (same sample should have same embedding)
        inv_loss = F.mse_loss(z1, z2)

        # === 2. Variance Loss ===
        # Force std of each dimension >= 1 (across batch)
        std_z1 = th.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = th.sqrt(z2.var(dim=0) + self.eps)
        var_loss = th.mean(F.relu(1 - std_z1)) + th.mean(F.relu(1 - std_z2))

        # === 3. Covariance Loss ===
        # Force off-diagonal elements of covariance matrix to be zero
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1)  # [D, D]
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1)  # [D, D]

        # Off-diagonal elements
        off_diag_mask = ~th.eye(D, dtype=th.bool, device=z1.device)
        cov_loss = (cov_z1[off_diag_mask].pow(2).sum() + cov_z2[off_diag_mask].pow(2).sum()) / D

        # === Total Loss ===
        total_loss = self.inv_weight * inv_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

        loss_dict = {
            "inv": inv_loss.item(),
            "var": var_loss.item(),
            "cov": cov_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict
