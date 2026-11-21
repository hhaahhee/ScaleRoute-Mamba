"""Named model variants shared by training and checkpoint evaluation."""

from copy import deepcopy


VARIANTS = (
    "full",
    "backbone",
    "encoder_only",
    "skip_only",
    "prior_only",
    "decoder_only",
    "uniform_route",
)


def model_config_for_variant(base_config, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    config = deepcopy(base_config)
    config.update(
        use_mbam=True,
        use_dual_skip=True,
        use_cross_stage_scale_fusion=True,
        use_encoder_scale_prior=True,
        use_decoder_scale_conditioning=True,
    )
    if variant == "backbone":
        config.update(use_mbam=False, use_dual_skip=False,
                      use_cross_stage_scale_fusion=False)
    elif variant == "encoder_only":
        config.update(use_dual_skip=False, use_cross_stage_scale_fusion=False)
    elif variant == "skip_only":
        config.update(use_mbam=False, use_cross_stage_scale_fusion=False)
    elif variant == "prior_only":
        config["use_decoder_scale_conditioning"] = False
    elif variant == "decoder_only":
        config["use_encoder_scale_prior"] = False
    elif variant == "uniform_route":
        config.update(use_encoder_scale_prior=False,
                      use_decoder_scale_conditioning=False)
    return config
