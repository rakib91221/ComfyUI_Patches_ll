import comfy
from .patch_util import set_hook, clean_hook, is_hunyuan_video_model, is_ltxv_video_model, is_mochi_video_model, is_wan_video_model
from .node_utils import get_new_forward_orig, get_old_method_name
from .patch_lib.WanVideoPatch import wan_forward


def video_outer_sample_function_wrapper(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                                  callback=None, disable_pbar=False, seed=None, **kwargs):
    cfg_guider = wrapper_executor.class_obj
    diffusion_model = cfg_guider.model_patcher.model.diffusion_model
    # set hook
    set_hook(diffusion_model, 'video_old_forward_orig', get_new_forward_orig(diffusion_model), get_old_method_name(diffusion_model))
    if is_wan_video_model(diffusion_model):
        # 原forward方法调用forward_origin时没有传transform_options，所以需要打补丁加上
        set_hook(diffusion_model, 'video_old_forward', wan_forward, "forward")

    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask, callback=callback,
                               disable_pbar=disable_pbar, seed=seed, **kwargs)
    finally:
        # cleanup hook
        clean_hook(diffusion_model, 'video_old_forward_orig', get_old_method_name(diffusion_model))
        if is_wan_video_model(diffusion_model):
            clean_hook(diffusion_model, 'video_old_forward', "forward")
    return out


class VideoForwardOverrider:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "patches/dit"
    DESCRIPTION = "Support HunYuanVideo"

    def apply_patch(self, model):
        model = model.clone()
        diffusion_model = model.get_model_object('diffusion_model')
        if is_hunyuan_video_model(diffusion_model) or is_ltxv_video_model(diffusion_model) or is_mochi_video_model(diffusion_model)\
                or is_wan_video_model(diffusion_model):
            patch_key = "video_forward_override_wrapper"
            if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) == 0:
                # Just add it once when connecting in series
                model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                           patch_key,
                                           video_outer_sample_function_wrapper
                                           )
        return (model,)


NODE_CLASS_MAPPINGS = {
    "VideoForwardOverrider": VideoForwardOverrider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoForwardOverrider": "VideoForwardOverrider",
}
