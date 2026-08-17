from experiments.model_clients import ModelClient

from .default import DefaultSeller


SELLER_CLASSES = {
    "default": DefaultSeller,
    "regulated_llm_seller": DefaultSeller,
}


def make_seller(
    seller_type: str,
    client: ModelClient,
    max_tokens: int,
    temperature: float,
    top_p: float,
    persona: str,
):
    try:
        cls = SELLER_CLASSES[seller_type]
    except KeyError as exc:
        raise ValueError(f"Unknown Simple_Env seller type: {seller_type}. Available: {sorted(SELLER_CLASSES)}") from exc
    return cls(client=client, max_tokens=max_tokens, temperature=temperature, top_p=top_p, persona=persona)
