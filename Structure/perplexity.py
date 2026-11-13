import torch

'''Implementing ADA conditional loss and perplexity scoring algorithm from:


    https://github.com/debatelab/ada-simulation


'''

def conditional_loss(labels, logits, k_last_tokens):

    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits = shift_logits[..., -k_last_tokens:, :].contiguous()
    shift_labels = shift_labels[..., -k_last_tokens:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss()

    last_k_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    return last_k_loss


def compute_perplexity(context, target, tokenizer, model):

    context_ids = tokenizer(
        context, add_special_tokens=False
    )["input_ids"]

    target_ids = tokenizer(
        target, add_special_tokens=False
    )["input_ids"]

    full_ids = context_ids + target_ids

    input_tensor = torch.tensor(
        [full_ids],
        device=model.device
    )

    with torch.inference_mode():
        outputs = model(input_ids=input_tensor)

        loss = conditional_loss(
            input_tensor,
            outputs.logits,
            len(target_ids)
        )

    return torch.exp(loss).item()


def compute_opinion(text, tokenizer, model):

    context = (
        "Let's discuss nuclear energy!\n"
        f"{text}\n"
        "I more or less agree with the arguments presented above. "
        "And therefore, all in all,"
    )

    pro = " nuclear energy should be expanded."
    con = " nuclear energy should be phased out."

    pro_ppl = compute_perplexity(context, pro, tokenizer, model)
    con_ppl = compute_perplexity(context, con, tokenizer, model)

    opinion = con_ppl / (pro_ppl + con_ppl)

    return opinion

