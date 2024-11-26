"""Benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""
from __future__ import annotations

import os
import logging

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

import mteb
from mteb import MTEB, get_tasks
import argparse
from dotenv import load_dotenv

load_dotenv() # for OPENAI

parser = argparse.ArgumentParser(description="Extract contexts")
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
    "MIRACLRetrieval",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "XPQARetrieval"
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

model_names = [
    # my_model_directory
]
model_names = [
    'Salesforce/SFR-Embedding-2_R', # 4096
    "Alibaba-NLP/gte-Qwen2-7B-instruct", # 8192
    "BAAI/bge-multilingual-gemma2", # 4096
    "intfloat/e5-mistral-7b-instruct", # 4096
    "intfloat/multilingual-e5-large-instruct", # 512
    "text-embedding-3-large", # 8191
    'Alibaba-NLP/gte-multilingual-base', 
    'intfloat/multilingual-e5-base', # 512
    'intfloat/multilingual-e5-large', # 512
    "jinaai/jina-embeddings-v3", # 8192
    "jhgan/ko-sroberta-multitask", # 128
    "BAAI/bge-m3", # 8192
    "nlpai-lab/KoE5", # 512
    "dragonkue/BGE-m3-ko" # 8192
] + model_names

def evaluate_model(model_name):
    try:
        model = None
        if not os.path.exists(model_name): # hf에 등록된 모델의 경우
            if "m2v" in model_name: # model2vec의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
                static_embedding = StaticEmbedding.from_model2vec(model_name)
                model = SentenceTransformer(modules=[static_embedding])
            else:
                model = mteb.get_model(model_name)
        else: # 직접 학습한 모델의 경우
            file_name = os.path.join(model_name, "model.safetensors")
            if os.path.exists(file_name):
                if "m2v" in model_name: # model2vec의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
                    static_embedding = StaticEmbedding.from_model2vec(model_name)
                    model = SentenceTransformer(modules=[static_embedding])
                else:
                    model = mteb.get_model(model_name)

        if model:
            print(f"Running task: {TASK_LIST} / {model_name}")
            evaluation = MTEB(
                tasks=get_tasks(tasks=TASK_LIST, languages=["kor-Kore", "kor-Hang", "kor_Hang"])
            )

            # 48GB VRAM 기준 적합한 batch sizes
            if "multilingual-e5" in model_name:
                batch_size = 256
            elif "jina" in model_name:
                batch_size = 16
            elif "bge" in model_name: # bge 기반의 모델들
                batch_size = 32
            elif "text-embedding" in model_name: # OPENAI Embedding은 VRAM과 상관없습니다.
                batch_size = 4096
            else:
                batch_size = 64
              
            evaluation.run(
                model,
                output_folder=f".results/{model_name}",
                encode_kwargs={"batch_size": batch_size},
            )
    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    for model_name in model_names:
        evaluate_model(model_name)
