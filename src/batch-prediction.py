import ray.data
import pandas as pd

model_id = "EleutherAI/gpt-j-6B"
revision = "float16"  # use float16 weights to fit in 16GB GPUs
prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

ray.init(
    runtime_env={
        "pip": [
            "accelerate>=0.16.0",
            "transformers>=4.26.0",
            "numpy<1.24",  # remove when mlflow updates beyond 2.2
            "torch",
        ]
    }
)

ds = ray.data.from_pandas(pd.DataFrame([prompt] * 10, columns=["prompt"]))

class PredictCallable:
    def __init__(self, model_id: str, revision: str = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",  # automatically makes use of all GPUs available to the Actor
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        tokenized = self.tokenizer(
            list(batch["prompt"]), return_tensors="pt"
        )
        input_ids = tokenized.input_ids.to(self.model.device)
        attention_mask = tokenized.attention_mask.to(self.model.device)

        gen_tokens = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.9,
            max_length=100,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return pd.DataFrame(
            self.tokenizer.batch_decode(gen_tokens), columns=["responses"]
        )
    
preds = (
    ds
    .repartition(100)
    .map_batches(
        PredictCallable,
        batch_size=8,
        fn_constructor_kwargs=dict(model_id=model_id, revision=revision),
        batch_format="pandas",
        compute=ray.data.ActorPoolStrategy(),
        num_gpus=0,
    )
)

preds.take_all()