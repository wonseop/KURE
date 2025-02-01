"""Benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""

from __future__ import annotations

import os
import logging
from multiprocessing import Process, current_process
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

import mteb
from mteb import MTEB, get_tasks
from mteb.encoder_interface import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.models.instruct_wrapper import instruct_wrapper

import argparse
from dotenv import load_dotenv
from setproctitle import setproctitle
import traceback
import logging
from datasets import load_dataset

load_dotenv()  # for OPENAI

parser = argparse.ArgumentParser(description="Extract contexts")
parser.add_argument("--quantize", default=False, type=bool, help="quantize embeddings")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = []

TASK_LIST_CLUSTERING = []

TASK_LIST_PAIR_CLASSIFICATION = []

TASK_LIST_RERANKING = []

TASK_LIST_RETRIEVAL = [
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "MIRACLRetrieval",  # 시간이 오래 걸림 주의
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MrTidyRetrieval",  # 시간이 오래 걸림 주의
    "MultiLongDocRetrieval",
    "XPQARetrieval",
    "Tatoeba",
]

TASK_LIST_STS = []

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

# MIRACL, MrTidy는 평가 시 시간이 오래 걸리기 때문에, 태스크별로 나누어 multiprocessing으로 평가합니다.
# 필요 시 GPU 번호를 다르게 조정해 주세요.
TASK_LIST_RETRIEVAL_GPU_MAPPING = {
    0: [
        "Ko-StrategyQA",
        "AutoRAGRetrieval",
        "PublicHealthQA",
        "BelebeleRetrieval",
        "XPQARetrieval",
        "MultiLongDocRetrieval",
    ],
    # 1: ["MIRACLRetrieval"],
    # 2: ["MrTidyRetrieval"],
}

model_names = [
    # my_model_directory
]
model_names = [
    "nlpai-lab/KURE-v1",  # 8192
    "elastic/multilingual-e5-small-optimized",  # 512
    "BAAI/bge-m3",  # 8192
    "intfloat/multilingual-e5-small",  # 512
    "dragonkue/BGE-m3-ko",  # 8192
    "intfloat/multilingual-e5-base",  # 512
    "nlpai-lab/KoE5",  # 512
    "intfloat/multilingual-e5-large",  # 512
    "intfloat/multilingual-e5-large-instruct",  # 512
    "ibm-granite/granite-embedding-107m-multilingual",  # 512
    "intfloat/e5-mistral-7b-instruct",  # 32768
    "ibm-granite/granite-embedding-278m-multilingual",  # 512
    # "Salesforce/SFR-Embedding-2_R",  # 4096
    # "Alibaba-NLP/gte-Qwen2-7B-instruct",  # 8192
    "BAAI/bge-multilingual-gemma2",  # 8192
    # "openai/text-embedding-3-large", # 8191
    "Alibaba-NLP/gte-multilingual-base",
    "jinaai/jina-embeddings-v3",  # 8192
    # "jhgan/ko-sroberta-multitask",  # 128
    "Snowflake/snowflake-arctic-embed-l-v2.0",  # 8192,
] + model_names


def evaluate_model(model_name, gpu_id, tasks):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        model = None
        if not os.path.exists(model_name):  # hf에 등록된 모델의 경우
            if (
                "m2v" in model_name
            ):  # model2vec의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
                static_embedding = StaticEmbedding.from_model2vec(model_name)
                model = SentenceTransformer(
                    modules=[static_embedding],
                    model_kwargs={"attn_implementation": "sdpa"},
                )
            else:
                if (
                    model_name == "nlpai-lab/KoE5"
                    or model_name == "KU-HIAI-ONTHEIT/ontheit-large-v1_1"
                ):
                    # mE5 기반의 모델이므로, 해당 프롬프트를 추가시킵니다.
                    model_prompts = {
                        PromptType.query.value: "query: ",
                        PromptType.passage.value: "passage: ",
                    }
                    model = SentenceTransformerWrapper(
                        model=model_name,
                        model_prompts=model_prompts,
                        model_kwargs={"attn_implementation": "sdpa"},
                    )
                elif model_name == "BAAI/bge-multilingual-gemma2":
                    # mbge-gemma2의 경우, mteb에서 지원하지 않습니다. 따라서, instruct_wrapper를 사용합니다.
                    instruction_template = "<instruct>{instruction}\n<query>"
                    model = instruct_wrapper(
                        model_name_or_path=model_name,
                        instruction_template=instruction_template,
                        attn="cccc",
                        pooling_method="lasttoken",
                        mode="embedding",
                        torch_dtype=torch.float16,
                        normalized=True,
                    )
                elif model_name == "Snowflake/snowflake-arctic-embed-l-v2.0":
                    # mteb에서 Snowflake 모델을 지원하지 않으므로, Snowflake에서 사용하는 "query: " prefix를 임의로 추가합니다.
                    model_prompts = {
                        PromptType.query.value: "query: ",
                    }
                    model = SentenceTransformerWrapper(
                        model=model_name,
                        model_prompts=model_prompts,
                        model_kwargs={"attn_implementation": "sdpa"},
                    )
                elif model_name == "Alibaba-NLP/gte-multilingual-base":
                    model = mteb.get_model(
                        model_name,
                        model_kwargs={"attn_implementation": "sdpa"},
                        trust_remote_code=True,
                    )
                else:
                    # mteb에 등록된 모델의 경우, 프롬프트/prefix 등을 포함하여 평가할 수 있습니다. 등록되지 않은 경우, sentence-transformer를 사용하여 불러옵니다.
                    model = mteb.get_model(
                        model_name,
                        model_kwargs={"attn_implementation": "sdpa"}
                    )
        else:  # 직접 학습한 모델의 경우
            file_name = os.path.join(model_name, "model.safetensors")
            if os.path.exists(file_name):
                if (
                    "m2v" in model_name
                ):  # model2vec의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
                    static_embedding = StaticEmbedding.from_model2vec(model_name)
                    model = SentenceTransformer(
                        modules=[static_embedding],
                        model_kwargs={"attn_implementation": "sdpa"},
                    )
                else:
                    model = mteb.get_model(
                        model_name, model_kwargs={"attn_implementation": "sdpa"}
                    )
    except Exception as ex:
        print("##### attention 적용 모델 로딩 실패 --> attention 적용 없이 재시도")
        print(ex)

        try:
            model = mteb.get_model(
                model_name
            )

        except Exception as ex:
            print(ex)
            traceback.print_exc()


    try:
        if model:
            setproctitle(f"{model_name}-{gpu_id}")
            print(
                f"Running tasks: {tasks} / {model_name} on GPU {gpu_id} in process {current_process().name}"
            )

            evaluation = MTEB(
                tasks=get_tasks(
                    tasks=tasks, languages=["kor-Kore", "kor-Hang", "kor_Hang"]
                )
            )
            # 48GB VRAM 기준 적합한 batch sizes
            if (
                "multilingual-e5" in model_name
                or "KoE5" in model_name
                or "ontheit" in model_name
                or "granite" in model_name
            ):
                batch_size = 512
            elif "jina" in model_name:
                batch_size = 8
            elif "bge-m3" in model_name or "Snowflake" in model_name:
                batch_size = 32
            elif "gemma2" in model_name:
                batch_size = 256
            elif "Salesforce" in model_name:
                batch_size = 128
            else:
                batch_size = 64

            if args.quantize:  # quantized model의 경우
                evaluation.run(
                    model,
                    output_folder=f"results/{model_name}-quantized",
                    encode_kwargs={"batch_size": batch_size, "precision": "binary"},
                )
            else:
                evaluation.run(
                    model,
                    output_folder=f"results/{model_name}",
                    encode_kwargs={"batch_size": batch_size},
                )
    except Exception as ex:
        print(ex)
        traceback.print_exc()


if __name__ == "__main__":
    processes = []

    print(f"taeminlee/Ko-StrategyQA loading")
    dataset = load_dataset('taeminlee/Ko-StrategyQA', 'default')

    # Evaluate one model at a time to better manage resources
    for model_name in model_names:

        print("\n\n")
        print("=" * 50)
        print(f"Running model: {model_name}")
        print("-" * 50)

        model_processes = []
        for gpu_id, tasks in TASK_LIST_RETRIEVAL_GPU_MAPPING.items():
            p = Process(target=evaluate_model, args=(model_name, gpu_id, tasks))
            p.start()
            model_processes.append(p)

        # Wait for all processes for current model to complete
        for p in model_processes:
            p.join()
