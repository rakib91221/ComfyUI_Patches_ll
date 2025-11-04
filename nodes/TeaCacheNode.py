import logging

import numpy as np
import torch.nn.functional as F

import comfy
from .patch_util import PatchKeys, add_model_patch_option, set_model_patch, set_model_patch_replace, \
    is_hunyuan_video_model, is_flux_model, is_ltxv_video_model, is_mochi_video_model, is_wan_video_model

tea_cache_key_attrs = "tea_cache_attr"
# https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py
# https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4HunyuanVideo/teacache_sample_video.py
# https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4LTX-Video/teacache_ltx.py
# https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Mochi/teacache_mochi.py
# https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Wan2.1/teacache_generate.py
coefficients_obj = {
    'Flux': [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    'HunYuanVideo': [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02],
    'LTXVideo': [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03],
    'MochiVideo': [-3.51241319e+03,  8.11675948e+02, -6.09400215e+01,  2.42429681e+00, 3.05291719e-03],
    # Supports 480P
    'WanVideo_t2v_1.3B': [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
    # Supports both 480P and 720P
    'WanVideo_t2v_14B': [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404],
    # Supports 480P
    'WanVideo_i2v_14B_480P': [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01],
    # Supports 720P
    'WanVideo_i2v_14B_720P': [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683],
    'WanVideo_disabled': [],
}

def get_teacache_global_cache(transformer_options, timesteps):
    diffusion_model = transformer_options.get(PatchKeys.running_net_model)
    if hasattr(diffusion_model, "flux_tea_cache"):
        tea_cache = getattr(diffusion_model, "flux_tea_cache", {})
        transformer_options[tea_cache_key_attrs] = tea_cache
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    attrs['step_i'] = timesteps[0].detach().cpu().item()
    # print(str(attrs['step_i']))

def tea_cache_enter_for_wanvideo(x, timestep, context, transformer_options, **kwargs):
    get_teacache_global_cache(transformer_options, timestep)
    return x, timestep, context

def tea_cache_enter_for_mochivideo(x, timestep, context, attention_mask, num_tokens, transformer_options, **kwargs):
    get_teacache_global_cache(transformer_options, timestep)
    return x, timestep, context, attention_mask, num_tokens

def tea_cache_enter_for_ltxvideo(x, timestep, context, attention_mask, frame_rate, guiding_latent, guiding_latent_noise_scale, transformer_options, **kwargs):
    get_teacache_global_cache(transformer_options, timestep)
    return x, timestep, context, attention_mask, frame_rate, guiding_latent, guiding_latent_noise_scale

# For Flux and HunYuanVideo
def tea_cache_enter(img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask, transformer_options, **kwargs):
    get_teacache_global_cache(transformer_options, timesteps)
    return img, img_ids, txt, txt_ids, timesteps, y, guidance, control, attn_mask

def tea_cache_patch_blocks_before(img, txt, vec, ids, pe, transformer_options, **kwargs):
    real_model = transformer_options[PatchKeys.running_net_model]
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    step_i = attrs['step_i']
    timestep_start = attrs['timestep_start']
    timestep_end = attrs['timestep_end']
    in_step = timestep_end <= step_i <= timestep_start
    # print(str(timestep_end)+' '+ str(step_i)+' '+str(timestep_start))

    # kijai版本TeaCache和TeaCache官方实现相结合在质量和速度上是最好的(即KJ-Nodes中的实现)
    # TeaCache官方实现只计算了cond的accumulated_rel_l1_distance，没有计算uncond的accumulated_rel_l1_distance
    accumulated_state = attrs.get('accumulated_state', {
        "x": {'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'skipped_steps': 0, 'previous_residual': None},
        'cond': {'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'skipped_steps': 0, 'previous_residual': None},
        'uncond': {'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'skipped_steps': 0, 'previous_residual': None}
    })
    teacache_enabled = attrs['rel_l1_thresh'] > 0 and in_step
    attrs['cache_enabled'] = teacache_enabled
    current_state_type = 'x'
    should_calc = True
    if teacache_enabled:
        inp = img
        vec_ = vec
        rescale_func_flag = True
        # split_cnd_flag=True是生效
        coefficient_type = 'Flux'
        if is_ltxv_video_model(real_model):
            coefficient_type = 'LTXVideo'
            modulated_inp = comfy.ldm.common_dit.rms_norm(inp)
            double_block_0 = real_model.transformer_blocks[0]
            num_ada_params = double_block_0.scale_shift_table.shape[0]
            ada_values = double_block_0.scale_shift_table[None, None] + vec_.reshape(img.shape[0], vec_.shape[1], num_ada_params, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
            modulated_inp = modulated_inp * (1 + scale_msa) + shift_msa
        elif is_mochi_video_model(real_model):
            coefficient_type = 'MochiVideo'
            double_block_0 = real_model.blocks[0]
            mod_x = double_block_0.mod_x(F.silu(vec_))
            scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)
            # copied from comfy.ldm.genmo.joint_model.asymm_models_joint.modulated_rmsnorm
            modulated_inp = comfy.ldm.common_dit.rms_norm(inp)
            modulated_inp = modulated_inp * (1 + scale_msa_x.unsqueeze(1))
        elif is_wan_video_model(real_model):
            coefficient_type = attrs.get("wan_coefficients_type", 'disabled')
            if coefficient_type == 'disabled':
                # e0
                rescale_func_flag = False
            else:
                vec_ = kwargs.get('e')
            modulated_inp = vec_
            coefficient_type = 'WanVideo_' + coefficient_type
            is_cond_flag = True if transformer_options["cond_or_uncond"] == [0] else False
            current_state_type = 'cond' if is_cond_flag else 'uncond'
        else:
            double_block_0 = real_model.double_blocks[0]
            img_mod1, img_mod2 = double_block_0.img_mod(vec_)
            modulated_inp = double_block_0.img_norm1(inp)
            if is_hunyuan_video_model(real_model):
                coefficient_type = 'HunYuanVideo'
                # if img_mod1.scale is None and img_mod1.shift is None:
                #     pass
                # elif img_mod1.shift is None:
                #     modulated_inp = modulated_inp * (1 + img_mod1.scale)
                # elif img_mod1.scale is None:
                #     modulated_inp =  modulated_inp + img_mod1.shift
                # else:
                #     modulated_inp = modulated_inp * (1 + img_mod1.scale) + img_mod1.shift
                if img_mod1.scale is not None:
                    modulated_inp = modulated_inp * (1 + img_mod1.scale)
                if img_mod1.shift is not None:
                    modulated_inp = modulated_inp + img_mod1.shift
            else:
                # Flux
                modulated_inp = (1 + img_mod1.scale) * modulated_inp + img_mod1.shift

        current_state = accumulated_state[current_state_type]

        if current_state.get('previous_modulated_input', None) is None or attrs['cnt'] == 0 or attrs['cnt'] == attrs['total_steps'] - 1:
            should_calc = True
            current_state['accumulated_rel_l1_distance'] = 0
        else:
            if rescale_func_flag:
                coefficients = coefficients_obj[coefficient_type]
                rescale_func = np.poly1d(coefficients)
                current_state['accumulated_rel_l1_distance'] += rescale_func(((modulated_inp - current_state['previous_modulated_input']).abs().mean() / current_state['previous_modulated_input'].abs().mean()).cpu().item())
            else:
                current_state['accumulated_rel_l1_distance'] += ((modulated_inp - current_state['previous_modulated_input']).abs().mean() / current_state['previous_modulated_input'].abs().mean()).cpu().item()

            if current_state['accumulated_rel_l1_distance'] < attrs['rel_l1_thresh']:
                should_calc = False
            else:
                should_calc = True
                current_state['accumulated_rel_l1_distance'] = 0

        current_state['previous_modulated_input'] = modulated_inp.clone().detach()

        attrs['cnt'] += 1
        if attrs['cnt'] == attrs['total_steps']:
            attrs['cnt'] = 0
        del inp, vec_
    else:
        # 设置了start_at场景需要初始化
        if is_wan_video_model(real_model):
            current_state_type = 'cond' if transformer_options["cond_or_uncond"] == [0] else 'uncond'

    attrs['should_calc'] = should_calc
    attrs['accumulated_state'] = accumulated_state
    attrs['current_state_type'] = current_state_type
    del real_model
    return img, txt, vec, ids, pe

def tea_cache_patch_double_blocks_replace(original_args, wrapper_options):
    img = original_args['img']
    txt = original_args['txt']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})

    should_calc = not attrs.get('cache_enabled') or attrs.get('should_calc', True)
    if not should_calc:
        current_state = attrs['accumulated_state'][attrs['current_state_type']]
        img += current_state['previous_residual'].to(img.device)
        current_state['skipped_steps'] += 1
    else:
        # (b, seq_len, _)
        attrs['ori_img'] = img.clone().detach()
        img, txt = wrapper_options.get('original_blocks')(**original_args, transformer_options=transformer_options)
    return img, txt

def tea_cache_patch_blocks_transition_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = not attrs.get('cache_enabled', False) or attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args, transformer_options=transformer_options)
    return img

def tea_cache_patch_single_blocks_replace(original_args, wrapper_options):
    img = original_args['img']
    txt = original_args['txt']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = not attrs.get('cache_enabled', False) or attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_blocks')(**original_args, transformer_options=transformer_options)
    return img, txt

def tea_cache_patch_blocks_after_replace(original_args, wrapper_options):
    img = original_args['img']
    transformer_options = wrapper_options.get('transformer_options', {})
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = not attrs.get('cache_enabled', False) or attrs.get('should_calc', True)
    if should_calc:
        img = wrapper_options.get('original_func')(**original_args)
    return img

def tea_cache_patch_final_transition_after(img, txt, transformer_options):
    attrs = transformer_options.get(tea_cache_key_attrs, {})
    should_calc = not attrs.get('cache_enabled', False) or attrs.get('should_calc', True)
    if should_calc:
        current_state = attrs['accumulated_state'][attrs['current_state_type']]
        current_state['previous_residual'] = (img - attrs['ori_img']).to(attrs['cache_device'])
    return img

def tea_cache_patch_dit_exit(img, transformer_options):
    tea_cache = transformer_options.get(tea_cache_key_attrs, {})
    setattr(transformer_options.get(PatchKeys.running_net_model), "flux_tea_cache", tea_cache)
    return img

def tea_cache_prepare_wrapper(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                                  callback=None, disable_pbar=False, seed=None, **kwargs):
    cfg_guider = wrapper_executor.class_obj

    # Use cfd_guider.model_options, which is copied from modelPatcher.model_options and will be restored after execution without any unexpected contamination
    temp_options = add_model_patch_option(cfg_guider, tea_cache_key_attrs)
    temp_options['total_steps'] = len(sigmas) - 1
    temp_options['cnt'] = 0
    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask, callback=callback,
                               disable_pbar=disable_pbar, seed=seed, **kwargs)
    finally:
        diffusion_model = cfg_guider.model_patcher.model.diffusion_model
        if hasattr(diffusion_model, "flux_tea_cache"):
            print_tea_cache_executed_state(getattr(diffusion_model, "flux_tea_cache"))
            del diffusion_model.flux_tea_cache

    return out

def print_tea_cache_executed_state(attrs):
    executed_state = attrs.get('accumulated_state', {})
    for state_type, state in executed_state.items():
        logging.info(f"skipped {state_type} steps: {state['skipped_steps']}")

class ApplyTeaCachePatchAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT",
                                  {
                                      "default": 0.25,
                                      "min": 0.0,
                                      "max": 5.0,
                                      "step": 0.001,
                                      "tooltip": "Flux: 0 (original), 0.25 (1.5x speedup), 0.4 (1.8x speedup), 0.6 (2.0x speedup), and 0.8 (2.25x speedup).\n"
                                                 "HunYuanVideo: 0 (original), 0.1 (1.6x speedup), 0.15 (2.1x speedup).\n"
                                                 "LTXVideo: 0 (original), 0.03 (1.6x speedup), 0.05 (2.1x speedup).\n"
                                                 "MochiVideo: 0 (original), 0.06 (1.5x speedup), 0.09 (2.1x speedup).\n"
                                                 "WanVideo: 0 (original), reference values\n"
                                                 "         Wan2.1 t2v 1.3B    0.05 0.07 0.08\n"
                                                 "         Wan2.1 t2v 14B    0.14 0.15 0.2\n"
                                                 "         Wan2.1 i2v 480P	0.13 0.19 0.26\n"
                                                 "         Wan2.1 i2v 720P	0.18 0.2 0.3"
                                  }),
                "start_at": ("FLOAT",
                             {
                                 "default": 0.0,
                                 "step": 0.01,
                                 "max": 1.0,
                                 "min": 0.0,
                             },
                             ),
                "end_at": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.01,
                    "max": 1.0,
                    "min": 0.0,
                }),
            },
            "optional": {
                "cache_device": (["main_device", "offload_device"], {"default": "offload_device"}),
                "wan_coefficients": (["disabled", "t2v_1.3B", "t2v_14B", "i2v_14B_480P", "i2v_14B_720P"], {
                    "default": "disabled",
                    "tooltip": "WanVideo coefficients."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch_advanced"
    CATEGORY = "patches/speed"
    DESCRIPTION = ("Apply the TeaCache patch to accelerate the model. Use it together with nodes that have the suffix ForwardOverrider."
                   "\nThis is effective only for Flux, HunYuanVideo, LTXVideo, WanVideo and MochiVideo.")

    def apply_patch_advanced(self, model, rel_l1_thresh, start_at=0.0, end_at=1.0, cache_device="offload_device", wan_coefficients="disabled", from_simple=False):

        model = model.clone()
        patch_key = "tea_cache_wrapper"
        if rel_l1_thresh == 0 or len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) > 0:
            return (model,)

        diffusion_model = model.get_model_object('diffusion_model')
        if not is_flux_model(diffusion_model) and not is_hunyuan_video_model(diffusion_model) and not is_ltxv_video_model(diffusion_model)\
                and not is_mochi_video_model(diffusion_model) and not is_wan_video_model(diffusion_model):
            logging.warning("TeaCache patch is not applied because the model is not supported.")
            return (model,)

        tea_cache_attrs = add_model_patch_option(model, tea_cache_key_attrs)

        tea_cache_attrs['rel_l1_thresh'] = rel_l1_thresh
        model_sampling = model.get_model_object("model_sampling")
        # For WanVideo, when wan_coefficients is disabled, the results of the first few steps are unstable?
        sigma_start = model_sampling.percent_to_sigma(max(start_at, 0.2) if from_simple and wan_coefficients == 'disabled' and is_wan_video_model(diffusion_model) else start_at)
        sigma_end = model_sampling.percent_to_sigma(end_at)
        tea_cache_attrs['timestep_start'] = model_sampling.timestep(sigma_start)
        tea_cache_attrs['timestep_end'] = model_sampling.timestep(sigma_end)
        tea_cache_attrs['cache_device'] = comfy.model_management.get_torch_device() if cache_device == "main_device" else comfy.model_management.unet_offload_device()

        if is_ltxv_video_model(diffusion_model):
            set_model_patch(model, PatchKeys.options_key, tea_cache_enter_for_ltxvideo, PatchKeys.dit_enter)
        elif is_mochi_video_model(diffusion_model):
            set_model_patch(model, PatchKeys.options_key, tea_cache_enter_for_mochivideo, PatchKeys.dit_enter)
        elif is_wan_video_model(diffusion_model):
            # i2v or t2v
            model_type = diffusion_model.model_type
            tea_cache_attrs['wan_coefficients_type'] = wan_coefficients
            if wan_coefficients != "disabled" and not wan_coefficients.startswith(model_type):
                logging.warning(f"The wan video's model type is {model_type}, but the selected wan_coefficients is {wan_coefficients}.")
            set_model_patch(model, PatchKeys.options_key, tea_cache_enter_for_wanvideo, PatchKeys.dit_enter)
        else:
            set_model_patch(model, PatchKeys.options_key, tea_cache_enter, PatchKeys.dit_enter)

        set_model_patch(model, PatchKeys.options_key, tea_cache_patch_blocks_before, PatchKeys.dit_blocks_before)

        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_double_blocks_replace, PatchKeys.dit_double_blocks_replace)
        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_blocks_transition_replace, PatchKeys.dit_blocks_transition_replace)
        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_single_blocks_replace, PatchKeys.dit_single_blocks_replace)
        set_model_patch_replace(model, PatchKeys.options_key, tea_cache_patch_blocks_after_replace, PatchKeys.dit_blocks_after_transition_replace)

        set_model_patch(model, PatchKeys.options_key, tea_cache_patch_final_transition_after, PatchKeys.dit_final_layer_before)
        set_model_patch(model, PatchKeys.options_key, tea_cache_patch_dit_exit, PatchKeys.dit_exit)

        # Just add it once when connecting in series
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                   patch_key,
                                   tea_cache_prepare_wrapper
                                   )
        return (model, )

class ApplyTeaCachePatch(ApplyTeaCachePatchAdvanced):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT",
                                  {
                                      "default": 0.25,
                                      "min": 0.0,
                                      "max": 5.0,
                                      "step": 0.001,
                                      "tooltip": "Flux: 0 (original), 0.25 (1.5x speedup), 0.4 (1.8x speedup), 0.6 (2.0x speedup), and 0.8 (2.25x speedup).\n"
                                                 "HunYuanVideo: 0 (original), 0.1 (1.6x speedup), 0.15 (2.1x speedup).\n"
                                                 "LTXVideo: 0 (original), 0.03 (1.6x speedup), 0.05 (2.1x speedup).\n"
                                                 "MochiVideo: 0 (original), 0.06 (1.5x speedup), 0.09 (2.1x speedup).\n"
                                                 "WanVideo: 0 (original), reference values\n"
                                                 "         Wan2.1 t2v 1.3B    0.05 0.07 0.08\n"
                                                 "         Wan2.1 t2v 14B    0.14 0.15 0.2\n"
                                                 "         Wan2.1 i2v 480P	0.13 0.19 0.26\n"
                                                 "         Wan2.1 i2v 720P	0.18 0.2 0.3"
                                  }),
            },
            "optional": {
                "cache_device": (["main_device", "offload_device"], {"default": "offload_device"}),
                "wan_coefficients": (["disabled", "t2v_1.3B", "t2v_14B", "i2v_14B_480P", "i2v_14B_720P"], {
                    "default": "disabled",
                    "tooltip": "WanVideo coefficients."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "patches/speed"
    DESCRIPTION = ("Apply the TeaCache patch to accelerate the model. Use it together with nodes that have the suffix ForwardOverrider."
                   "\nThis is effective only for Flux, HunYuanVideo, LTXVideo, WanVideo and MochiVideo.")

    def apply_patch(self, model, rel_l1_thresh, cache_device="offload_device", wan_coefficients="disabled"):

        return super().apply_patch_advanced(model, rel_l1_thresh, start_at=0.0, end_at=1.0, cache_device=cache_device, wan_coefficients=wan_coefficients, from_simple=False)

NODE_CLASS_MAPPINGS = {
    "ApplyTeaCachePatch": ApplyTeaCachePatch,
    "ApplyTeaCachePatchAdvanced": ApplyTeaCachePatchAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyTeaCachePatch": "ApplyTeaCachePatch",
    "ApplyTeaCachePatchAdvanced": "ApplyTeaCachePatchAdvanced",
}
