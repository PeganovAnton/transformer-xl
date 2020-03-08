import torch
from torch import nn
from torch.nn import functional as F, NLLLoss
from transformers import TransfoXLPreTrainedModel, TransfoXLModel
from transformers.modeling_transfo_xl_utilities import LogUniformSampler, ProjectedAdaptiveLogSoftmax, sample_logits
from typing import Tuple


class TransfoXLLMHeadModelFixed(TransfoXLPreTrainedModel):
    """Fixed Transformer-XL with LM Head model.
    The fixes are:
        1) DataParallel friendly
        2) Returns average of losses, not (bsz, seq_len)
        3) Can return loss and scores at the same time (return_scores parameter in forward)
    """

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TransfoXLModel(config)
        self.sample_softmax = config.sample_softmax
        # use sampled softmax
        if config.sample_softmax > 0:
            self.out_layer = nn.Linear(config.d_model, config.vocab_size)
            self.sampler = LogUniformSampler(config.vocab_size, config.sample_softmax)
        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(
                config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val
            )
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """
        Run this to be sure output and input (adaptive) softmax weights are tied
        """
        # sampled softmax
        if self.sample_softmax > 0:
            if self.config.tie_weight:
                self.out_layer.weight = self.transformer.word_emb.weight
        # adaptive softmax (including standard softmax)
        else:
            if self.config.tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self._tie_or_clone_weights(self.crit.out_layers[i], self.transformer.word_emb.emb_layers[i])
            if self.config.tie_projs:
                for i, tie_proj in enumerate(self.config.tie_projs):
                    if tie_proj and self.config.div_val == 1 and self.config.d_model != self.config.d_embed:
                        if self.config.torchscript:
                            self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[0].clone())
                        else:
                            self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[0]
                    elif tie_proj and self.config.div_val != 1:
                        if self.config.torchscript:
                            self.crit.out_projs[i] = nn.Parameter(self.transformer.word_emb.emb_projs[i].clone())
                        else:
                            self.crit.out_projs[i] = self.transformer.word_emb.emb_projs[i]

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.transformer.reset_length(tgt_len, ext_len, mem_len)

    def init_mems(self, bsz):
        return self.transformer.init_mems(bsz)

    def forward(
        self,
        *mems: torch.Tensor,
        input_ids: torch.Tensor = None,
        head_mask: Tuple[torch.Tensor] = None,
        inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_scores: bool = False,
    ):
        """Args:
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems
            given to this model should not be passed as input ids as they have already been computed.
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        head_mask (:obj:`Tuple[torch.FloatTensor]` with one tensor of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        return_scores (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to return scores, if labels are provided. (Will compute PAS twice)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.TransfoXLConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.'
        mems (:obj:`config.n_layers` tensors: :obj:`torch.FloatTensor`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.

    Examples::

        from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
        import torch

        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, mems = outputs[:2]

        """
        if head_mask:
            assert isinstance(head_mask, tuple) and len(head_mask) == 1, (
                f"You have to provide head_mask in an 1-element tuple to prevent scattering into different GPUs. "
                f"Got: {head_mask}"
            )
            head_mask = head_mask[0]

        if input_ids is not None:
            bsz, tgt_len = input_ids.size(0), input_ids.size(1)
        elif inputs_embeds is not None:
            bsz, tgt_len = inputs_embeds.size(0), inputs_embeds.size(1)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert all(mem.size(0) == bsz for mem in mems)
        if mems:
            # (bsz, ...) -> original shape
            mems = [mem.transpose(0, 1) for mem in mems]
        else:
            # Turn an empty tuple to None
            mems = None

        transformer_outputs = self.transformer(input_ids, mems=mems, head_mask=head_mask, inputs_embeds=inputs_embeds)

        last_hidden = transformer_outputs[0]
        pred_hid = last_hidden[:, -tgt_len:]

        # From original shape (..., bsz, ...) -> (bsz, ...)
        new_mems = [mem.transpose(0, 1) for mem in transformer_outputs[1]]

        assert all(mem.size(0) == bsz for mem in new_mems)

        outputs = transformer_outputs[2:]
        if self.sample_softmax > 0 and self.training:
            assert self.config.tie_weight
            logit = sample_logits(self.transformer.word_emb, self.out_layer.bias, labels, pred_hid, self.sampler)
            softmax_output = -F.log_softmax(logit, -1)[:, :, 0]
            outputs = [softmax_output] + outputs + new_mems
            if labels is not None:
                # TODO: This is not implemented
                raise NotImplementedError
        else:
            # Adaptive softmax
            if labels is not None and not return_scores:
                # Here fast PAS NLL loss computations without intermediate scores
                loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), labels).mean()
                outputs = [loss, None] + outputs + new_mems
            else:
                # Should return scores
                scores = self.crit(pred_hid.view(-1, pred_hid.size(-1)), None).view(
                    bsz, tgt_len, -1
                )  # log softmax scores
                outputs = [scores] + outputs + new_mems
                if labels is not None:
                    loss_function = NLLLoss()
                    loss = loss_function(scores.view(-1, scores.size(-1)), labels.view(-1))
                    outputs = [loss] + outputs

        # (loss), (logits), (all hidden states), (all attentions), new_mems
        return outputs

    def get_output_embeddings(self):
        """ Double-check if you are using adaptive softmax.
        """
        if self.sample_softmax > 0:
            return self.out_layer
        else:
            return self.crit.out_layers[-1]

    def prepare_inputs_for_generation(self, input_ids, past, **model_kwargs):
        inputs = {"input_ids": input_ids}

        # if past is defined in model kwargs then use it for faster decoding
        if past:
            inputs["mems"] = past

        return inputs
