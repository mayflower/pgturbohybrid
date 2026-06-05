#!/usr/bin/env python3
"""Rewrite pg_colbert_v1 ModernColBERT GGUFs for llama.cpp's BERT loader."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

TYPE_UINT8 = 0
TYPE_INT8 = 1
TYPE_UINT16 = 2
TYPE_INT16 = 3
TYPE_UINT32 = 4
TYPE_INT32 = 5
TYPE_FLOAT32 = 6
TYPE_BOOL = 7
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_UINT64 = 10
TYPE_INT64 = 11
TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

SCALAR_FORMATS = {
    TYPE_UINT8: "<B",
    TYPE_INT8: "<b",
    TYPE_UINT16: "<H",
    TYPE_INT16: "<h",
    TYPE_UINT32: "<I",
    TYPE_INT32: "<i",
    TYPE_FLOAT32: "<f",
    TYPE_BOOL: "<?",
    TYPE_UINT64: "<Q",
    TYPE_INT64: "<q",
    TYPE_FLOAT64: "<d",
}


@dataclass
class Kv:
    key: str
    typ: int
    value: Any


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    typ: int
    offset: int


@dataclass
class TensorOut:
    name: str
    dims: list[int]
    typ: int
    data: bytes


@dataclass
class ProjectionOut:
    dims: list[int]
    typ: int
    data: bytes


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def read_string(buf: bytes, pos: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", buf, pos)
    pos += 8
    raw = buf[pos : pos + length]
    pos += length
    return raw.decode("utf-8"), pos


def write_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def read_value(buf: bytes, pos: int, typ: int) -> tuple[Any, int]:
    if typ == TYPE_STRING:
        return read_string(buf, pos)
    if typ == TYPE_ARRAY:
        array_type, length = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        values = []
        for _ in range(length):
            value, pos = read_value(buf, pos, array_type)
            values.append(value)
        return (array_type, values), pos
    fmt = SCALAR_FORMATS[typ]
    (value,) = struct.unpack_from(fmt, buf, pos)
    return value, pos + struct.calcsize(fmt)


def write_value(typ: int, value: Any) -> bytes:
    if typ == TYPE_STRING:
        return write_string(value)
    if typ == TYPE_ARRAY:
        array_type, values = value
        out = bytearray(struct.pack("<IQ", array_type, len(values)))
        for item in values:
            out += write_value(array_type, item)
        return bytes(out)
    return struct.pack(SCALAR_FORMATS[typ], value)


def tensor_nbytes(tensor: TensorInfo) -> int:
    count = 1
    for dim in tensor.dims:
        count *= dim
    if tensor.typ == GGML_TYPE_F32:
        return count * 4
    if tensor.typ == GGML_TYPE_F16:
        return count * 2
    raise ValueError(f"unsupported tensor type {tensor.typ} for {tensor.name}")


def f16_data_to_f32(data: bytes) -> bytes:
    out = bytearray(len(data) * 2)
    pos = 0
    for (value,) in struct.iter_unpack("<e", data):
        struct.pack_into("<f", out, pos, float(value))
        pos += 4
    return bytes(out)


def convert_tensor_data(tensor: TensorInfo, data: bytes, output_f32: bool) -> tuple[int, bytes]:
    if not output_f32:
        return tensor.typ, data
    if tensor.typ == GGML_TYPE_F32:
        return tensor.typ, data
    if tensor.typ == GGML_TYPE_F16:
        return GGML_TYPE_F32, f16_data_to_f32(data)
    raise ValueError(f"unsupported tensor type {tensor.typ} for {tensor.name}")


def parse_gguf(path: Path) -> tuple[list[Kv], list[TensorInfo], int, int, bytes]:
    buf = path.read_bytes()
    if buf[:4] != GGUF_MAGIC:
        raise ValueError(f"{path} is not a GGUF file")
    version, tensor_count, kv_count = struct.unpack_from("<IQQ", buf, 4)
    if version != GGUF_VERSION:
        raise ValueError(f"unsupported GGUF version {version}")

    pos = 4 + 4 + 8 + 8
    kvs = []
    alignment = 32
    for _ in range(kv_count):
        key, pos = read_string(buf, pos)
        (typ,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        value, pos = read_value(buf, pos, typ)
        kvs.append(Kv(key, typ, value))
        if key == "general.alignment":
            alignment = int(value)

    tensors = []
    for _ in range(tensor_count):
        name, pos = read_string(buf, pos)
        (n_dims,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        dims = list(struct.unpack_from("<" + "Q" * n_dims, buf, pos))
        pos += 8 * n_dims
        typ, offset = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        tensors.append(TensorInfo(name, dims, typ, offset))

    data_start = align(pos, alignment)
    return kvs, tensors, data_start, alignment, buf


def kv_map(kvs: list[Kv]) -> dict[str, Kv]:
    return {kv.key: kv for kv in kvs}


def set_kv(kvs: list[Kv], key: str, typ: int, value: Any) -> None:
    for kv in kvs:
        if kv.key == key:
            kv.typ = typ
            kv.value = value
            return
    kvs.append(Kv(key, typ, value))


def get_int(kvs_by_key: dict[str, Kv], key: str) -> int:
    try:
        return int(kvs_by_key[key].value)
    except KeyError as exc:
        raise ValueError(f"missing required GGUF key {key}") from exc


def get_float(kvs_by_key: dict[str, Kv], key: str) -> float:
    try:
        return float(kvs_by_key[key].value)
    except KeyError as exc:
        raise ValueError(f"missing required GGUF key {key}") from exc


def llama_wpm_token(token: str, token_type: int) -> str:
    if token_type != 1:
        return token
    if token.startswith("##"):
        return token[2:]
    return "\u2581" + token


def tokenizer_from_hf_json(kvs_by_key: dict[str, Kv]) -> tuple[list[str], list[int]]:
    try:
        tokenizer_json = json.loads(kvs_by_key["tokenizer.huggingface.json"].value)
    except KeyError as exc:
        raise ValueError("missing tokenizer.huggingface.json") from exc

    vocab = tokenizer_json.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer.huggingface.json does not contain model.vocab")
    max_id = max(vocab.values())
    for added in tokenizer_json.get("added_tokens", []):
        if isinstance(added, dict) and isinstance(added.get("id"), int):
            max_id = max(max_id, added["id"])

    tokens = [None] * (max_id + 1)
    token_types = [1] * (max_id + 1)
    for token, idx in vocab.items():
        if not isinstance(idx, int) or idx < 0 or idx >= len(tokens):
            raise ValueError(f"invalid tokenizer vocab id for {token!r}: {idx!r}")
        tokens[idx] = token
    for added in tokenizer_json.get("added_tokens", []):
        if not isinstance(added, dict):
            continue
        idx = added.get("id")
        content = added.get("content")
        if not isinstance(idx, int) or not isinstance(content, str):
            continue
        if idx < 0 or idx >= len(tokens):
            raise ValueError(f"invalid added tokenizer id for {content!r}: {idx!r}")
        tokens[idx] = content
        token_types[idx] = 3 if added.get("special") else 4

    missing = [str(i) for i, token in enumerate(tokens) if token is None]
    if missing:
        raise ValueError("tokenizer vocab ids are not contiguous: " + ", ".join(missing[:10]))
    return [
        llama_wpm_token(str(token), token_type)
        for token, token_type in zip(tokens, token_types)
    ], token_types


def tensor_rename(name: str) -> str | None:
    fixed = {
        "hf.embeddings.word_embeddings.weight": "token_embd.weight",
        "hf.embeddings.token_type_embeddings.weight": "token_types.weight",
        "hf.embeddings.position_embeddings.weight": "position_embd.weight",
        "hf.embeddings.LayerNorm.weight": "token_embd_norm.weight",
        "hf.embeddings.LayerNorm.bias": "token_embd_norm.bias",
    }
    if name in fixed:
        return fixed[name]

    prefix = "hf.encoder.layer."
    if not name.startswith(prefix):
        return None

    rest = name[len(prefix) :]
    layer, _, suffix = rest.partition(".")
    if not layer.isdigit() or not suffix:
        return None

    suffix_map = {
        "attention.self.query.weight": "blk.{layer}.attn_q.weight",
        "attention.self.query.bias": "blk.{layer}.attn_q.bias",
        "attention.self.key.weight": "blk.{layer}.attn_k.weight",
        "attention.self.key.bias": "blk.{layer}.attn_k.bias",
        "attention.self.value.weight": "blk.{layer}.attn_v.weight",
        "attention.self.value.bias": "blk.{layer}.attn_v.bias",
        "attention.output.dense.weight": "blk.{layer}.attn_output.weight",
        "attention.output.dense.bias": "blk.{layer}.attn_output.bias",
        "attention.output.LayerNorm.weight": "blk.{layer}.attn_output_norm.weight",
        "attention.output.LayerNorm.bias": "blk.{layer}.attn_output_norm.bias",
        "intermediate.dense.weight": "blk.{layer}.ffn_up.weight",
        "intermediate.dense.bias": "blk.{layer}.ffn_up.bias",
        "output.dense.weight": "blk.{layer}.ffn_down.weight",
        "output.dense.bias": "blk.{layer}.ffn_down.bias",
        "output.LayerNorm.weight": "blk.{layer}.layer_output_norm.weight",
        "output.LayerNorm.bias": "blk.{layer}.layer_output_norm.bias",
    }
    template = suffix_map.get(suffix)
    if template is None:
        return None
    return template.format(layer=layer)


def build_output(
    kvs: list[Kv],
    tensors: list[TensorInfo],
    data_start: int,
    buf: bytes,
    output_f32: bool,
) -> tuple[list[Kv], list[TensorOut], ProjectionOut]:
    by_key = kv_map(kvs)
    tokens, token_types = tokenizer_from_hf_json(by_key)

    out_kvs = [Kv(kv.key, kv.typ, kv.value) for kv in kvs]
    set_kv(out_kvs, "bert.context_length", TYPE_UINT32, get_int(by_key, "bert.max_position_embeddings"))
    set_kv(out_kvs, "bert.embedding_length", TYPE_UINT32, get_int(by_key, "bert.hidden_size"))
    set_kv(out_kvs, "bert.feed_forward_length", TYPE_UINT32, get_int(by_key, "bert.intermediate_size"))
    set_kv(out_kvs, "bert.block_count", TYPE_UINT32, get_int(by_key, "bert.num_hidden_layers"))
    set_kv(out_kvs, "bert.attention.head_count", TYPE_UINT32, get_int(by_key, "bert.num_attention_heads"))
    set_kv(out_kvs, "bert.attention.layer_norm_epsilon", TYPE_FLOAT32, get_float(by_key, "bert.layer_norm_eps"))
    set_kv(out_kvs, "tokenizer.ggml.model", TYPE_STRING, "bert")
    set_kv(out_kvs, "tokenizer.ggml.tokens", TYPE_ARRAY, (TYPE_STRING, tokens))
    set_kv(out_kvs, "tokenizer.ggml.token_type", TYPE_ARRAY, (TYPE_INT32, token_types))
    set_kv(out_kvs, "tokenizer.ggml.token_type_count", TYPE_UINT32, get_int(by_key, "bert.type_vocab_size"))
    set_kv(out_kvs, "tokenizer.ggml.bos_token_id", TYPE_UINT32, get_int(by_key, "colbert.cls_token_id"))
    set_kv(out_kvs, "tokenizer.ggml.seperator_token_id", TYPE_UINT32, get_int(by_key, "colbert.sep_token_id"))
    set_kv(out_kvs, "tokenizer.ggml.padding_token_id", TYPE_UINT32, get_int(by_key, "colbert.pad_token_id"))
    set_kv(out_kvs, "tokenizer.ggml.unknown_token_id", TYPE_UINT32, 100)
    set_kv(out_kvs, "tokenizer.ggml.mask_token_id", TYPE_UINT32, 103)
    set_kv(out_kvs, "tokenizer.ggml.add_bos_token", TYPE_BOOL, True)
    set_kv(out_kvs, "tokenizer.ggml.add_eos_token", TYPE_BOOL, False)
    set_kv(out_kvs, "tokenizer.ggml.add_sep_token", TYPE_BOOL, True)

    out_tensors = []
    projection = None
    seen = set()
    for tensor in tensors:
        new_name = tensor_rename(tensor.name)
        start = data_start + tensor.offset
        end = start + tensor_nbytes(tensor)
        if tensor.name == "colbert.proj.weight":
            projection = ProjectionOut(tensor.dims, tensor.typ, buf[start:end])
            continue
        if new_name is None:
            continue
        if new_name in seen:
            raise ValueError(f"duplicate output tensor {new_name}")
        seen.add(new_name)
        typ, data = convert_tensor_data(tensor, buf[start:end], output_f32)
        out_tensors.append(TensorOut(new_name, tensor.dims, typ, data))
    if projection is None:
        raise ValueError("missing required tensor colbert.proj.weight")
    return out_kvs, out_tensors, projection


def write_gguf(path: Path, kvs: list[Kv], tensors: list[TensorOut], alignment: int) -> None:
    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(kvs))
    for kv in kvs:
        header += write_string(kv.key)
        header += struct.pack("<I", kv.typ)
        header += write_value(kv.typ, kv.value)

    data = bytearray()
    tensor_infos = bytearray()
    for tensor in tensors:
        offset = align(len(data), alignment)
        data += b"\0" * (offset - len(data))
        tensor_infos += write_string(tensor.name)
        tensor_infos += struct.pack("<I", len(tensor.dims))
        tensor_infos += struct.pack("<" + "Q" * len(tensor.dims), *tensor.dims)
        tensor_infos += struct.pack("<IQ", tensor.typ, offset)
        data += tensor.data

    out = header + tensor_infos
    out += b"\0" * (align(len(out), alignment) - len(out))
    out += data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)


def write_projection_sidecar(path: Path, projection: ProjectionOut) -> None:
    if len(projection.dims) != 2:
        raise ValueError("ColBERT projection tensor must be 2-dimensional")
    out = bytearray()
    out += b"PGCPROJ1"
    out += struct.pack("<IQQ", projection.typ, projection.dims[0], projection.dims[1])
    out += projection.data
    path.write_bytes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--preserve-tensor-types",
        action="store_true",
        help="keep source tensor types instead of writing CPU-compatible f32 tensors",
    )
    args = parser.parse_args()

    kvs, tensors, data_start, alignment, buf = parse_gguf(args.input)
    out_kvs, out_tensors, projection = build_output(
        kvs,
        tensors,
        data_start,
        buf,
        output_f32=not args.preserve_tensor_types,
    )
    write_gguf(args.output, out_kvs, out_tensors, alignment)
    sidecar = args.output.with_suffix(args.output.suffix + ".colbert_proj")
    write_projection_sidecar(sidecar, projection)
    print(f"wrote {args.output} with {len(out_tensors)} tensors")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
