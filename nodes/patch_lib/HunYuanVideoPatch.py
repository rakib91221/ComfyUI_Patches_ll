import torch
from torch import Tensor

from ..patch_util import PatchKeys
from comfy.ldm.flux.layers import timestep_embedding


def hunyuan_forward_orig(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    txt_mask: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor = None,
    guiding_frame_index=None,
    control=None,
    transformer_options={},
    **kwargs
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    patches_point = transformer_options.get(PatchKeys.options_key, {})

    transformer_options[PatchKeys.running_net_model] = self

    patches_enter = patches_point.get(PatchKeys.dit_enter, [])
    if patches_enter is not None and len(patches_enter) > 0:
        for patch_enter in patches_enter:
            img, img_ids, txt, txt_ids, timesteps, y, guidance, control, txt_mask = patch_enter(img,
                                                                                                img_ids,
                                                                                                txt,
                                                                                                txt_ids,
                                                                                                timesteps,
                                                                                                y,
                                                                                                guidance,
                                                                                                control,
                                                                                                attn_mask=txt_mask,
                                                                                                transformer_options=transformer_options
                                                                                                )

    initial_shape = list(img.shape)
    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

    if guiding_frame_index is not None:
        token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
        vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
        vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
        frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
        modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
        modulation_dims_txt = [(0, None, 1)]
    else:
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
        modulation_dims = None
        modulation_dims_txt = None

    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if txt_mask is not None and not torch.is_floating_point(txt_mask):
        txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

    txt = self.txt_in(txt, timesteps, txt_mask)

    ids = torch.cat((img_ids, txt_ids), dim=1)
    pe = self.pe_embedder(ids)

    img_len = img.shape[1]
    if txt_mask is not None:
        attn_mask_len = img_len + txt.shape[1]
        attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
        attn_mask[:, 0, img_len:] = txt_mask
    else:
        attn_mask = None

    blocks_replace = patches_replace.get("dit", {})

    patch_blocks_before = patches_point.get(PatchKeys.dit_blocks_before, [])
    if patch_blocks_before is not None and len(patch_blocks_before) > 0:
        for blocks_before in patch_blocks_before:
            img, txt, vec, ids, pe = blocks_before(img, txt, vec, ids, pe, transformer_options)

    def double_blocks_wrap(img, txt, vec, pe, control=None, attn_mask=None, transformer_options={},
                           modulation_dims_img=None, modulation_dims_txt=None):
        running_net_model = transformer_options[PatchKeys.running_net_model]
        patch_double_blocks_with_control_replace = patches_point.get(PatchKeys.dit_double_block_with_control_replace)
        for i, block in enumerate(running_net_model.double_blocks):
            if patch_double_blocks_with_control_replace is not None:
                img, txt = patch_double_blocks_with_control_replace({'i': i,
                                                                     'block': block,
                                                                     'img': img,
                                                                     'txt': txt,
                                                                     'vec': vec,
                                                                     'pe': pe,
                                                                     'control': control,
                                                                     'attn_mask': attn_mask,
                                                                     'modulation_dims_img': modulation_dims_img,
                                                                     'modulation_dims_txt': modulation_dims_txt
                                                                     },
                                                                    {
                                                                        "original_func": double_block_and_control_replace,
                                                                        "transformer_options": transformer_options
                                                                    })
            else:
                img, txt = double_block_and_control_replace(i=i,
                                                            block=block,
                                                            img=img,
                                                            txt=txt,
                                                            vec=vec,
                                                            pe=pe,
                                                            control=control,
                                                            attn_mask=attn_mask,
                                                            modulation_dims_img=modulation_dims_img,
                                                            modulation_dims_txt=modulation_dims_txt,
                                                            transformer_options=transformer_options
                                                            )

        del patch_double_blocks_with_control_replace
        return img, txt

    patch_double_blocks_replace = patches_point.get(PatchKeys.dit_double_blocks_replace)

    if patch_double_blocks_replace is not None:
        img, txt = patch_double_blocks_replace({"img": img,
                                                "txt": txt,
                                                "vec": vec,
                                                "pe": pe,
                                                "control": control,
                                                "attn_mask": attn_mask,
                                                "modulation_dims_img": modulation_dims,
                                                "modulation_dims_txt": modulation_dims_txt,
                                                },
                                               {
                                                   "original_blocks": double_blocks_wrap,
                                                   "transformer_options": transformer_options
                                               })
    else:
        img, txt = double_blocks_wrap(img=img,
                                      txt=txt,
                                      vec=vec,
                                      pe=pe,
                                      control=control,
                                      attn_mask=attn_mask,
                                      modulation_dims_img=modulation_dims,
                                      modulation_dims_txt=modulation_dims_txt,
                                      transformer_options=transformer_options
                                      )

    patches_double_blocks_after = patches_point.get(PatchKeys.dit_double_blocks_after, [])
    if patches_double_blocks_after is not None and len(patches_double_blocks_after) > 0:
        for patch_double_blocks_after in patches_double_blocks_after:
            img, txt = patch_double_blocks_after(img, txt, transformer_options)

    patch_blocks_transition = patches_point.get(PatchKeys.dit_blocks_transition_replace)

    def blocks_transition_wrap(**kwargs):
        txt = kwargs["txt"]
        img = kwargs["img"]
        return torch.cat((img, txt), 1)

    if patch_blocks_transition is not None:
        img = patch_blocks_transition({"img": img, "txt": txt, "vec": vec, "pe": pe},
                                      {
                                          "original_func": blocks_transition_wrap,
                                          "transformer_options": transformer_options
                                      })
    else:
        img = blocks_transition_wrap(img=img, txt=txt)

    patches_single_blocks_before = patches_point.get(PatchKeys.dit_single_blocks_before, [])
    if patches_single_blocks_before is not None and len(patches_single_blocks_before) > 0:
        for patch_single_blocks_before in patches_single_blocks_before:
            img, txt = patch_single_blocks_before(img, txt, transformer_options)

    def single_blocks_wrap(img, txt, vec, pe, control=None, attn_mask=None, transformer_options={}, modulation_dims=None):
        running_net_model = transformer_options[PatchKeys.running_net_model]
        for i, block in enumerate(running_net_model.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       attn_mask=args.get("attention_mask"),
                                       modulation_dims=args.get("modulation_dims"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "attention_mask": attn_mask,
                                                           'modulation_dims': modulation_dims},
                                                          {
                                                              "original_block": block_wrap,
                                                              "transformer_options": transformer_options
                                                          })
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims)

            if control is not None:  # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, : img_len] += add

        return img

    patch_single_blocks_replace = patches_point.get(PatchKeys.dit_single_blocks_replace)

    if patch_single_blocks_replace is not None:
        img, txt = patch_single_blocks_replace({"img": img,
                                                "txt": txt,
                                                "vec": vec,
                                                "pe": pe,
                                                "control": control,
                                                "attn_mask": attn_mask,
                                                "modulation_dims": modulation_dims,
                                                },
                                               {
                                                   "original_blocks": single_blocks_wrap,
                                                   "transformer_options": transformer_options
                                               })
    else:
        img = single_blocks_wrap(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 control=control,
                                 attn_mask=attn_mask,
                                 modulation_dims=modulation_dims,
                                 transformer_options=transformer_options
                                 )

    patch_blocks_exit = patches_point.get(PatchKeys.dit_blocks_after, [])
    if patch_blocks_exit is not None and len(patch_blocks_exit) > 0:
        for blocks_after in patch_blocks_exit:
            img, txt = blocks_after(img, txt, transformer_options)

    def final_transition_wrap(**kwargs):
        img = kwargs["img"]
        img_len = kwargs["img_len"]
        return img[:, : img_len]

    patch_blocks_after_transition_replace = patches_point.get(PatchKeys.dit_blocks_after_transition_replace)
    if patch_blocks_after_transition_replace is not None:
        img = patch_blocks_after_transition_replace({"img": img, "txt": txt, "vec": vec, "pe": pe, "img_len": img_len},
                                                    {
                                                        "original_func": final_transition_wrap,
                                                        "transformer_options": transformer_options
                                                    })
    else:
        img = final_transition_wrap(img=img, img_len=img_len)

    patches_final_layer_before = patches_point.get(PatchKeys.dit_final_layer_before, [])
    if patches_final_layer_before is not None and len(patches_final_layer_before) > 0:
        for patch_final_layer_before in patches_final_layer_before:
            img = patch_final_layer_before(img, txt, transformer_options)

    img = self.final_layer(img, vec, modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

    shape = initial_shape[-3:]
    for i in range(len(shape)):
        shape[i] = shape[i] // self.patch_size[i]
    img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
    img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
    img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])

    patches_exit = patches_point.get(PatchKeys.dit_exit, [])
    if patches_exit is not None and len(patches_exit) > 0:
        for patch_exit in patches_exit:
            img = patch_exit(img, transformer_options)

    del transformer_options[PatchKeys.running_net_model]

    return img


def double_block_and_control_replace(i, block, img, txt=None, vec=None, pe=None, control=None, attn_mask=None, transformer_options={}, modulation_dims_img=None, modulation_dims_txt=None):
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    if ("double_block", i) in blocks_replace:
        def block_wrap(args):
            out = {}
            out["img"], out["txt"] = block(img=args["img"],
                                           txt=args["txt"],
                                           vec=args["vec"],
                                           pe=args["pe"],
                                           attn_mask=args.get("attention_mask"),
                                           modulation_dims_img=args["modulation_dims_img"],
                                           modulation_dims_txt=args["modulation_dims_txt"])
            return out

        out = blocks_replace[("double_block", i)]({"img": img,
                                                   "txt": txt,
                                                   "vec": vec,
                                                   "pe": pe,
                                                   "attention_mask": attn_mask,
                                                   'modulation_dims_img': modulation_dims_img,
                                                   'modulation_dims_txt': modulation_dims_txt
                                                   },
                                                  {
                                                      "original_block": block_wrap,
                                                      "transformer_options": transformer_options
                                                  })
        txt = out["txt"]
        img = out["img"]
    else:
        img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims_img, modulation_dims_txt=modulation_dims_txt)
    if control is not None:  # Controlnet
        control_i = control.get("input")
        if i < len(control_i):
            add = control_i[i]
            if add is not None:
                img += add

    return img, txt
