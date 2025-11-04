import logging

import comfy
from .patch_util import PatchKeys, add_model_patch_option, set_model_patch, set_model_patch_replace, \
    is_hunyuan_video_model, is_flux_model, is_ltxv_video_model, is_mochi_video_model, is_wan_video_model

fb_cache_key_attrs = "fb_cache_attr"
fb_cache_model_temp = "flux_fb_cache"

def get_fb_cache_global_cache(transformer_options, timesteps):
    diffusion_model = transformer_options.get(PatchKeys.running_net_model)
    if hasattr(diffusion_model, fb_cache_model_temp):
        tea_cache = getattr(diffusion_model, fb_cache_model_temp, {})
        transformer_options[fb_cache_key_attrs] = tea_cache

    attrs = transformer_options.get(fb_cache_key_attrs, {})
    attrs['step_i'] = timesteps[0].detach().cpu().item()

def fb_cache_enter_for_wanvideo(x, timestep, context, transformer_options):
    get_fb_cache_global_cache(transformer_options, timestep)
    return x, timestep, context

def fb_cache_enter_for_mochivideo(x, timestep, context, attention_mask, num_tokens, transformer_options):
    get_fb_cache_global_cache(transformer_options, timestep)
    return x, timestep, context, attention_mask, num_tokens

def fb_cache_enter_for_ltxvideo(x, timestep, context, attention_mask, frame_rate, guiding_latent, guiding_latent_noise_scale, transformer_options):
    get_fb_cache_global_cache(transformer_options, timestep)
    return x, timestep, context, attention_mask, frame_rate, guiding_latent, guiding_latent_noise_scale

# For Flux and HunYuanVideo
def fb_cache_enter(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options):
    get_fb_cache_global_cache(transformer_options, timesteps)
    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask

def are_two_tensors_similar(t1, t2, *, threshold):
    if t1.shape != t2.shape:
        return False
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    diff = mean_diff / mean_t1
    return diff.item() < threshold

def fb_cache_patch_double_block_with_control_replace(original_args, wrapper_options):
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    step_i = attrs['step_i']
    timestep_start = attrs['timestep_start']
    timestep_end = attrs['timestep_end']
    in_step = timestep_end <= step_i <=timestep_start
    if not in_step:
        attrs['should_calc'] = True
        return wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)

    block_i = original_args['i']
    txt = original_args['txt']
    if block_i == 0:
        # 与上一次采样中的first double block输出比较，绝对均值差值小于threshold则可以使用缓存
        img, txt = wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)

        previous_first_block_residual = attrs.get('previous_first_block_residual')
        if previous_first_block_residual is not None:
            should_calc = not are_two_tensors_similar(previous_first_block_residual, img, threshold=attrs['rel_diff_threshold'])
        else:
            # 需要计算，即：不使用缓存
            should_calc = True

        if should_calc:
            attrs['previous_first_block_residual'] = img.clone()
        else:
            # 上次非缓存采样值
            previous_residual = attrs.get('previous_residual')
            if previous_residual is not None:
                img += previous_residual

        attrs['should_calc'] = should_calc
        attrs['ori_img'] = None
    else:
        img = original_args['img']
        should_calc = attrs['should_calc']
        if should_calc:
            if attrs['ori_img'] is None:
                attrs['ori_img'] = original_args['img'].clone()
            if block_i > 0:
                img, txt = wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)

    del attrs, transformer_options
    return img, txt

def fb_cache_patch_blocks_transition_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)
    return img

def fb_cache_patch_single_blocks_replace(original_args, wrapper_options):
    img = original_args['img']
    txt = original_args['txt']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_blocks')(**original_args, transformer_options=transformer_options)
    return img, txt

def fb_cache_patch_blocks_after_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args)
    return img

def fb_cache_patch_final_transition_after(img, txt, transformer_options):
    attrs = transformer_options.get(fb_cache_key_attrs, {})
    should_calc = attrs.get('should_calc', True)
    if should_calc:
        if attrs.get('ori_img') is not None:
            attrs['previous_residual'] = img - attrs['ori_img']
    return img

def fb_cache_patch_dit_exit(img, transformer_options):
    tea_cache = transformer_options.get(fb_cache_key_attrs, {})
    setattr(transformer_options.get(PatchKeys.running_net_model), fb_cache_model_temp, tea_cache)
    return img

def fb_cache_prepare_wrapper(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                                  callback=None, disable_pbar=False, seed=None, **kwargs):
    cfg_guider = wrapper_executor.class_obj

    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask, callback=callback,
                               disable_pbar=disable_pbar, seed=seed, **kwargs)
    finally:
        diffusion_model = cfg_guider.model_patcher.model.diffusion_model
        if hasattr(diffusion_model, fb_cache_model_temp):
            delattr(diffusion_model, fb_cache_model_temp)

    return out

class ApplyFirstBlockCachePatchAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "residual_diff_threshold": ("FLOAT",
                                            {
                                                "default": 0.00,
                                                "min": 0.0,
                                                "max": 1.0,
                                                "step": 0.01,
                                                "tooltip": "Flux: 0 (original), 0.12 (1.8x speedup).\n"
                                                           "HunYuanVideo: 0 (original), 0.1 (1.6x speedup).\n"
                                                           "LTXVideo: 0 (original), 0.5 (1.2x speedup).\n"
                                                           "MochiVideo: 0 (original), 0.03 (1.5x speedup).\n"
                                                           "WanVideo: 0 (original), 0.05 (1.5x speedup)."
                                            }),
                "start_at": ("FLOAT",
                             {
                                 "default": 0.0,
                                 "step": 0.01,
                                 "max": 1.0,
                                 "min": 0.0
                             }
                             ),
                "end_at": ("FLOAT",
                           {
                               "default": 1.0,
                               "step": 0.01,
                               "max": 1.0,
                               "min": 0.0
                           })
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch_advanced"
    CATEGORY = "patches/speed"
    DESCRIPTION = ("Apply the First Block Cache patch to accelerate the model. Use it together with nodes that have the suffix ForwardOverrider."
                   "\nThis is effective only for Flux, HunYuanVideo, LTXVideo, WanVideo and MochiVideo.")

    def apply_patch_advanced(self, model, residual_diff_threshold, start_at=0.0, end_at=1.0):

        model = model.clone()
        patch_key = "fb_cache_wrapper"
        if residual_diff_threshold == 0 or len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) > 0:
            return (model,)

        diffusion_model = model.get_model_object('diffusion_model')
        if not is_flux_model(diffusion_model) and not is_hunyuan_video_model(diffusion_model) and not is_ltxv_video_model(diffusion_model)\
                and not is_mochi_video_model(diffusion_model) and not is_wan_video_model(diffusion_model):
            logging.warning("First Block Cache patch is not applied because the model is not supported.")
            return (model,)

        fb_cache_attrs = add_model_patch_option(model, fb_cache_key_attrs)

        fb_cache_attrs['rel_diff_threshold'] = residual_diff_threshold
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_at)
        sigma_end = model_sampling.percent_to_sigma(end_at)
        fb_cache_attrs['timestep_start'] = model_sampling.timestep(sigma_start)
        fb_cache_attrs['timestep_end'] = model_sampling.timestep(sigma_end)

        if is_ltxv_video_model(diffusion_model):
            set_model_patch(model, PatchKeys.options_key, fb_cache_enter_for_ltxvideo, PatchKeys.dit_enter)
        elif is_mochi_video_model(diffusion_model):
            set_model_patch(model, PatchKeys.options_key, fb_cache_enter_for_mochivideo, PatchKeys.dit_enter)
        elif is_wan_video_model(diffusion_model):
            set_model_patch(model, PatchKeys.options_key, fb_cache_enter_for_wanvideo, PatchKeys.dit_enter)
        else:
            set_model_patch(model, PatchKeys.options_key, fb_cache_enter, PatchKeys.dit_enter)

        set_model_patch_replace(model, PatchKeys.options_key, fb_cache_patch_double_block_with_control_replace, PatchKeys.dit_double_block_with_control_replace)
        set_model_patch_replace(model, PatchKeys.options_key, fb_cache_patch_blocks_transition_replace, PatchKeys.dit_blocks_transition_replace)
        set_model_patch_replace(model, PatchKeys.options_key, fb_cache_patch_single_blocks_replace, PatchKeys.dit_single_blocks_replace)
        set_model_patch_replace(model, PatchKeys.options_key, fb_cache_patch_blocks_after_replace, PatchKeys.dit_blocks_after_transition_replace)

        set_model_patch(model, PatchKeys.options_key, fb_cache_patch_final_transition_after, PatchKeys.dit_final_layer_before)
        set_model_patch(model, PatchKeys.options_key, fb_cache_patch_dit_exit, PatchKeys.dit_exit)

        # Just add it once when connecting in series
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                   patch_key,
                                   fb_cache_prepare_wrapper
                                   )
        return (model, )

class ApplyFirstBlockCachePatch(ApplyFirstBlockCachePatchAdvanced):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "residual_diff_threshold": ("FLOAT",
                                            {
                                                "default": 0.00,
                                                "min": 0.0,
                                                "max": 1.0,
                                                "step": 0.01,
                                                "tooltip": "Flux: 0 (original), 0.12 (1.8x speedup).\n"
                                                           "HunYuanVideo: 0 (original), 0.1 (1.6x speedup).\n"
                                                           "LTXVideo: 0 (original), 0.05 (1.2x speedup).\n"
                                                           "MochiVideo: 0 (original), 0.03 (1.5x speedup).\n"
                                                           "WanVideo: 0 (original), 0.05 (1.5x speedup)."
                                            })
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "patches/speed"
    DESCRIPTION = ("Apply the First Block Cache patch to accelerate the model. Use it together with nodes that have the suffix ForwardOverrider."
                   "\nThis is effective only for Flux, HunYuanVideo, LTXVideo, WanVideo and MochiVideo.")

    def apply_patch(self, model, residual_diff_threshold):
        return super().apply_patch_advanced(model, residual_diff_threshold, start_at=0.0, end_at=1.0)

NODE_CLASS_MAPPINGS = {
    "ApplyFirstBlockCachePatch": ApplyFirstBlockCachePatch,
    "ApplyFirstBlockCachePatchAdvanced": ApplyFirstBlockCachePatchAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyFirstBlockCachePatch": "ApplyFirstBlockCachePatch",
    "ApplyFirstBlockCachePatchAdvanced": "ApplyFirstBlockCachePatchAdvanced",
}
