"""Add action tokens and four structural special tokens before Qwen3-VL SFT."""

import argparse
from pathlib import Path
import shutil


SPECIAL_TOKENS = ["<thinking>", "</thinking>", "<answer>", "</answer>"]
REFERENCE_FOR_THINK = "explanation"
REFERENCE_FOR_ANSWER = "PT"


def add_action_tokens(tokenizer, num_action_tokens):
    """Preserve action tokens during decoding and check adjacent joint tokens."""
    if type(num_action_tokens) is not int or num_action_tokens <= 0:
        raise ValueError("num_action_tokens must be a positive integer")
    from transformers import AddedToken

    tokens = [f"<action_{i}>" for i in range(num_action_tokens)]
    added = tokenizer.add_tokens(
        [AddedToken(token, single_word=True, normalized=False) for token in tokens],
        special_tokens=False,
    )
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(set(ids)) != len(tokens) or any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise ValueError("Action tokens do not have unique vocabulary IDs")
    if set(ids) & set(tokenizer.all_special_ids):
        raise ValueError("Action tokens must be ordinary tokens")
    for token, token_id in zip(tokens, ids):
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Action token is not atomic: {token}")
    if tokenizer.encode("".join(tokens), add_special_tokens=False) != ids:
        raise ValueError("Adjacent action tokens are not encoded atomically")
    if tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != "".join(tokens):
        raise ValueError("Action tokens do not survive ordinary decoding")
    return added, ids


def validate_structure_tokens(tokenizer):
    """Check that all four tags are atomic special tokens and decode intact."""
    ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if len(set(ids)) != len(SPECIAL_TOKENS) or any(i is None or i == tokenizer.unk_token_id for i in ids):
        raise ValueError("Structural tokens do not have unique vocabulary IDs")
    if not set(ids).issubset(tokenizer.all_special_ids):
        raise ValueError("Structural tokens must be registered as special tokens")
    for token, token_id in zip(SPECIAL_TOKENS, ids):
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Structural token is not atomic: {token}")
    if tokenizer.encode("".join(SPECIAL_TOKENS), add_special_tokens=False) != ids:
        raise ValueError("Adjacent structural tokens are not encoded atomically")
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != "".join(SPECIAL_TOKENS):
        raise ValueError("Structural tokens are not preserved during decoding")
    return ids


def add_structure_tokens(tokenizer):
    """Append tags while preserving existing special tokens and vocabulary IDs."""
    existing_vocab = tokenizer.get_vocab()
    existing_special_ids = set(tokenizer.all_special_ids)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS},
        replace_additional_special_tokens=False,
    )
    ids = validate_structure_tokens(tokenizer)
    if not existing_special_ids.issubset(tokenizer.all_special_ids):
        raise ValueError("Existing model special tokens were replaced")
    new_tokens = [token for token in SPECIAL_TOKENS if token not in existing_vocab]
    return added, ids, new_tokens


def initialize_structure_embeddings(model, tokenizer, new_tokens):
    """Initialize only new tag rows in both input embeddings and an untied head."""
    if not new_tokens:
        return
    import torch

    reference_for_token = dict(zip(SPECIAL_TOKENS, [
        REFERENCE_FOR_THINK, REFERENCE_FOR_THINK, REFERENCE_FOR_ANSWER, REFERENCE_FOR_ANSWER,
    ]))
    input_embeddings = model.get_input_embeddings()
    layers = [input_embeddings]
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
        layers.append(output_embeddings)

    reference_ids = {}
    for reference in set(reference_for_token[token] for token in new_tokens):
        ids = tokenizer.encode(reference, add_special_tokens=False)
        if not ids or any(i == tokenizer.unk_token_id or i < 0 or
                          any(i >= layer.weight.shape[0] for layer in layers) for i in ids):
            raise ValueError(f"Cannot encode embedding reference: {reference}")
        reference_ids[reference] = ids

    with torch.no_grad():
        for layer in layers:
            references = {reference: layer.weight[ids].mean(dim=0) for reference, ids in reference_ids.items()}
            for token in new_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                layer.weight[token_id].copy_(references[reference_for_token[token]])


def copy_remaining_files(source, destination):
    """Copy all remaining top-level files without overwriting saved files."""
    for source_file in source.iterdir():
        if source_file.is_file():
            target = destination / source_file.name
            if not target.exists():
                print(f"Copying remaining file: {source_file.name}")
                shutil.copy2(source_file, target)


def extend_vocab(base_model_path, output_path, num_action_tokens, dtype="bfloat16"):
    """Save a complete checkpoint with resized input and output embeddings."""
    source, destination = Path(base_model_path).resolve(), Path(output_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Local base model not found: {source}")
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Input and output model directories must be separate")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Output directory must be empty: {destination}")
    if type(num_action_tokens) is not int or num_action_tokens <= 0:
        raise ValueError("num_action_tokens must be a positive integer")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported model dtype")

    import torch
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    print(f"Loading tokenizer and processor: {source}")
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True)
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
    old_size = len(tokenizer)
    added, token_ids = add_action_tokens(tokenizer, num_action_tokens)
    special_added, special_ids, new_special_tokens = add_structure_tokens(tokenizer)
    print(f"Action tokens: requested={num_action_tokens}, added={added}")
    print(f"Structural special tokens: added={special_added}, tokens={SPECIAL_TOKENS}")
    print(f"Vocabulary: {old_size} -> {len(tokenizer)}")

    print(f"Loading model ({dtype}): {source}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source, trust_remote_code=True, dtype=getattr(torch, dtype),
    )
    model.resize_token_embeddings(len(tokenizer))
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        raise ValueError("Input embeddings do not match the tokenizer")
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings.weight.shape[0] != len(tokenizer):
        raise ValueError("Output embeddings do not match the tokenizer")
    initialize_structure_embeddings(model, tokenizer, new_special_tokens)
    print(f"Initialized {len(new_special_tokens)} new structural tokens from explanation/PT references")

    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    processor.tokenizer = tokenizer
    processor.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    copy_remaining_files(source, destination)

    saved = AutoTokenizer.from_pretrained(destination, trust_remote_code=True, use_fast=True)
    if saved.convert_tokens_to_ids([f"<action_{i}>" for i in range(num_action_tokens)]) != token_ids:
        raise ValueError("Saved tokenizer changed action token IDs")
    if validate_structure_tokens(saved) != special_ids:
        raise ValueError("Saved tokenizer changed structural token IDs")
    print(f"Model and multimodal processor saved to: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Local Qwen3-VL model directory")
    parser.add_argument("--output-dir", required=True, help="New, empty model directory")
    parser.add_argument("--num-action-tokens", type=int, required=True, help="Total action vocabulary size")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()
    extend_vocab(args.base_model, args.output_dir, args.num_action_tokens, args.dtype)


if __name__ == "__main__":
    main()
