import comfy
from .node_utils import get_old_method_name, get_new_forward_orig
from .patch_util import set_hook, clean_hook, is_flux_model

def flux_outer_sample_function_wrapper(wrapper_executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                                  callback=None, disable_pbar=False, seed=None, **kwargs):
    cfg_guider = wrapper_executor.class_obj
    diffusion_model = cfg_guider.model_patcher.model.diffusion_model
    # set hook
    set_hook(diffusion_model, 'flux_old_forward_orig', get_new_forward_orig(diffusion_model), get_old_method_name(diffusion_model))

    try:
        out = wrapper_executor(noise, latent_image, sampler, sigmas, denoise_mask=denoise_mask, callback=callback,
                               disable_pbar=disable_pbar, seed=seed, **kwargs)
    finally:
        # cleanup hook
        clean_hook(diffusion_model, 'flux_old_forward_orig')
    return out


class FluxForwardOverrider:

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

    def apply_patch(self, model):

        model = model.clone()
        if is_flux_model(model.get_model_object('diffusion_model')):
            patch_key = "flux_forward_override_wrapper"
            if len(model.get_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, patch_key)) == 0:
                # Just add it once when connecting in series
                model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                           patch_key,
                                           flux_outer_sample_function_wrapper
                                           )
        return (model, )


NODE_CLASS_MAPPINGS = {
    "FluxForwardOverrider": FluxForwardOverrider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxForwardOverrider": "FluxForwardOverrider",
}
