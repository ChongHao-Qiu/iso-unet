"""
models.py — Model factory.

Dynamically import and instantiate models via the `model_class` string in config
(e.g. 'iso_unet.baseline_canet.CANetBaseline'). Each model also declares its
`dataset_kind` (single_frame / sequence) so train.py / eval.py can pick the
right dataset.

Integration with iso_unet: all supported models are already Lightning Modules
(with forward / training_step / configure_optimizers etc.), so model_kwargs
can be passed straight to __init__.
"""
import importlib
import torch


# ── Default dataset_kind per model class (config can override) ──
DEFAULT_DATASET_KIND = {
    # Single-frame baselines
    'iso_unet.baseline_unet2d.UNet2DBaseline':                    'single_frame',
    'iso_unet.baseline_unet2d_vanilla.UNet2DVanillaBaseline':     'single_frame',
    'iso_unet.baseline_unetpp.UNetPlusPlusBaseline':              'single_frame',
    'iso_unet.baseline_attunet.AttentionUNetBaseline':            'single_frame',
    'iso_unet.baseline_dcsaunet.DCSAUNetBaseline':                'single_frame',
    'iso_unet.baseline_transunet.TransUNetBaseline':              'single_frame',
    'iso_unet.baseline_uno.UNOBaseline':                          'single_frame',
    'iso_unet.unet_v2.UNet2DBaseline':                            'single_frame',
    'iso_unet.model.FNO2d':                                       'single_frame',
    'iso_unet.baseline_segformer.SegFormerBaseline':              'single_frame',
    'iso_unet.baseline_canet.CANetBaseline':                      'single_frame',
    'iso_unet.baseline_ufno_temporal.UFNOTemporalBaseline':       'single_frame',  # T=1 default
    'iso_unet.ddpm_unet.DDPMLightning':                           'single_frame',
    # Sequence baselines
    'iso_unet.baseline_unet3d_temporal.TemporalUNet3DBaseline':   'sequence',
    'iso_unet.baseline_iso_unet.IsoUNetBaseline':       'sequence',
    'iso_unet.baseline_iso_unet_v2.IsoUNetBaselineV2':  'sequence',
    'iso_unet.convlstm.ConvLSTMSeq':                              'sequence',
}


def _import_class(dotted_path):
    """e.g. 'iso_unet.baseline_canet.CANetBaseline' → the class object."""
    module_path, class_name = dotted_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_model(model_class_path, model_kwargs=None, weights=None):
    """
    Args:
        model_class_path: str like 'iso_unet.baseline_canet.CANetBaseline'
        model_kwargs:     dict of __init__ kwargs (n_inputs, base_channels, ...)
        weights:          optional lat_weights tensor; injected if class accepts 'weights'.

    Automatically filters out kwargs that class.__init__ doesn't recognize
    (e.g. FNO2d has no lr parameter), printing a warning but not erroring —
    the model runs with its own defaults.

    Returns:
        L.LightningModule instance
    """
    import inspect
    cls = _import_class(model_class_path)
    kw  = dict(model_kwargs or {})

    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if 'weights' in params and weights is not None and 'weights' not in kw:
        kw['weights'] = weights

    # Filter out kwargs the class can't accept (unless it has **kwargs)
    if not accepts_var_kw:
        valid = set(params.keys())
        dropped = [k for k in kw if k not in valid]
        if dropped:
            print(f'  [build_model] {cls.__name__} does not accept these kwargs; filtered out: {dropped}')
        kw = {k: v for k, v in kw.items() if k in valid}

    instance = cls(**kw)

    # Fix legacy models that store `weights` as plain attribute (e.g. FNO2d):
    # convert to buffer so it moves with .cuda() / .to(device).
    import torch
    if isinstance(getattr(instance, 'weights', None), torch.Tensor):
        already_buffer = ('weights' in instance._buffers)
        if not already_buffer:
            w = instance.weights
            delattr(instance, 'weights')
            instance.register_buffer('weights', w.clone())

    return instance


def get_dataset_kind(model_class_path, override=None):
    """Get dataset_kind for a model. Config can override the default."""
    if override is not None:
        return override
    if model_class_path in DEFAULT_DATASET_KIND:
        return DEFAULT_DATASET_KIND[model_class_path]
    raise ValueError(f'Unknown model class "{model_class_path}". '
                     f'Either add to DEFAULT_DATASET_KIND in models.py or specify '
                     f'`dataset_kind` in config.')


def resolve_model_kwargs(model_kwargs, input_features, lr=None, verbose=True):
    """Fill in the model kwargs that depend on the channel ordering of `input_features`.

    Configs leave `landfrac_idx` / `pr_channels` / `ice_idx` unset (or stale) because
    they are properties of the chosen input_set, not of the model. This resolves them
    from `input_features` so that train.py and any standalone script build exactly the
    same model from the same YAML.

    Returns a new dict; the input is not mutated.
    """
    kw = dict(model_kwargs or {})
    kw.setdefault('n_inputs', len(input_features))
    if lr is not None:
        kw.setdefault('lr', lr)

    def _log(msg):
        if verbose:
            print(msg)

    # landfrac_idx — correct it if the config disagrees with input_features
    if 'landfrac_idx' in kw and 'LANDFRAC' in input_features:
        expected_lf = input_features.index('LANDFRAC')
        if kw['landfrac_idx'] != expected_lf:
            _log(f'  [auto] landfrac_idx: {kw["landfrac_idx"]} \u2192 {expected_lf} '
                 f'(LANDFRAC at position {expected_lf} in {input_features})')
            kw['landfrac_idx'] = expected_lf

    # pr_channels — which input channels to sum into the precipitation routing mask
    if kw.get('use_precip_routing') and not kw.get('pr_channels'):
        if 'pr' in input_features:
            kw['pr_channels'] = (input_features.index('pr'),)
        elif 'precl' in input_features and 'precc' in input_features:
            kw['pr_channels'] = (input_features.index('precl'),
                                 input_features.index('precc'))
        else:
            raise ValueError(
                f'use_precip_routing=True but neither "pr" nor ("precl","precc") in input_features: '
                f'{input_features}. Add them to the config or set pr_channels manually.')
        _log(f'  [auto] pr_channels: {kw["pr_channels"]}  (auto-detected from input_features)')

    # ice_idx — sea-ice fraction channel for the ice expert
    if kw.get('use_ice_routing') and kw.get('ice_idx') is None:
        if 'aice' in input_features:
            kw['ice_idx'] = input_features.index('aice')
            _log(f'  [auto] ice_idx: {kw["ice_idx"]}  (aice at position {kw["ice_idx"]} in input_features)')
        else:
            raise ValueError(
                f'use_ice_routing=True but "aice" not in input_features: {input_features}. '
                f'Add aice to the input_set (e.g. use full_split) or set ice_idx manually.')

    return kw


def supports_attention_viz(model):
    """Whether the model has a forward_with_attn method (for --save_attn)."""
    return hasattr(model, 'forward_with_attn') and callable(model.forward_with_attn)


# ── Inference utility — unified for single_frame / sequence models ──
@torch.no_grad()
def run_inference_unified(model, test_input, device, batch_size=8,
                          dataset_kind='single_frame'):
    """
    Run prediction, handling 4 cases uniformly:

    dataset_kind='single_frame':
        test_input = (eval_x, eval_y)  tuple of tensors (N_eval, C/1, H, W)
        Model input (B, C, H, W) → output (B, 1, H, W).
        Returns: truth (N, H, W), pred (N, H, W)

    dataset_kind='sequence':
        test_input = test_ds  (SequenceDataset)
        Model input (B, T, C, H, W) → output is either (B, T, 1, H, W) (dense, e.g. ConvLSTMSeq)
                                      or (B, 1, H, W) (last-frame only, e.g. IsoUNet)
        Returns: truth (N, H, W), pred (N, H, W) — taking the last frame of each window
    """
    model.eval()
    truths, preds = [], []

    if dataset_kind == 'single_frame':
        eval_x, eval_y = test_input
        n = len(eval_x)
        for i in range(0, n, batch_size):
            xb = eval_x[i:i+batch_size].to(device)
            yb = eval_y[i:i+batch_size]
            y_hat = model(xb)
            # Some models (e.g. DDPM) may return a tuple — unwrap if so
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]
            truths.append(yb.cpu())
            preds .append(y_hat.cpu())
        truth = torch.cat(truths).squeeze(1).numpy()
        pred  = torch.cat(preds).squeeze(1).numpy()
        return truth, pred

    elif dataset_kind == 'sequence':
        loader = torch.utils.data.DataLoader(
            test_input, batch_size=batch_size, shuffle=False, num_workers=0,
        )
        for batch in loader:
            x, y_seq = batch[0].to(device), batch[1]
            y_hat = model(x)
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]
            # If model outputs dense sequence (B, T, 1, H, W), take last frame
            if y_hat.dim() == 5:
                y_hat = y_hat[:, -1]            # (B, 1, H, W)
            # GT y_seq: (B, T, 1, H, W), take last frame
            y_last = y_seq[:, -1]
            truths.append(y_last.cpu())
            preds .append(y_hat.cpu())
        truth = torch.cat(truths).squeeze(1).numpy()
        pred  = torch.cat(preds).squeeze(1).numpy()
        return truth, pred

    else:
        raise ValueError(f"dataset_kind must be 'single_frame' or 'sequence', got {dataset_kind}")
