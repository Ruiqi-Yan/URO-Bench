import transformers
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
import argparse
import os
import jsonlines
import logging


class MultiTurnAssistant:
    def __init__(self, whisper_path, llm_path):
        self.whisper_path = whisper_path
        self.llm_path = llm_path
        self.asr = self.load_asr()
        self.llm = self.load_llm()

    def load_asr(self):
        model_id = self.whisper_path
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir="./cache",
        )
        model.to("cuda:0")
        processor = AutoProcessor.from_pretrained(model_id)
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda:0",
        )
        return asr_pipe

    def load_llm(self):
        model_id = self.llm_path
        llm_pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda",
        )
        return llm_pipe

    def transcribe_audio(self, audio_path: str) -> str:
        audio, sr = sf.read(audio_path)
        result = self.asr(audio, generate_kwargs={"return_timestamps": True})
        transcript = result["text"].strip()
        return transcript

    def chat(self, conversation):
        outputs = self.llm(
            conversation,
            max_new_tokens=2048,
        )

        generated_text = outputs[0]["generated_text"][-1]["content"]
        return generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["audio", "text"],
        help="Input modality: 'audio' or 'text'",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--whisper_path", type=str, required=True)
    parser.add_argument("--llm_path", type=str, required=True)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # inference
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_text = os.path.join(output_dir, "output_with_text.jsonl")

    logging.info("<========loading model========>")
    assistant = MultiTurnAssistant(args.whisper_path, args.llm_path)

    logging.info("<========inference starts========>")
    with open(args.dataset, "r") as f, jsonlines.open(output_text, mode="w") as ot:
        for data in jsonlines.Reader(f):
            dialogue = data["dialogue"]
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who tries to help answer the user's question. Please note that the user's query is transcribed from speech, and the transcription may contain errors.",
                }
            ]

            last_response = ""
            conversation = []
            for turn in dialogue:
                input_path = os.path.join(
                    os.path.dirname(args.dataset), turn["source_wav"]
                )
                user_transcript = assistant.transcribe_audio(input_path)

                messages.append({"role": "user", "content": user_transcript})

                assistant_response = assistant.chat(messages)
                last_response = assistant_response.strip()

                messages.append({"role": "assistant", "content": last_response})
                conversation.append(
                    {
                        "round": turn["round"],
                        "source_wav": turn["source_wav"],
                        "source_text": turn["source_text"],
                        "target_text": turn["target_text"],
                        "output_text": last_response,
                    }
                )

            output_data = {
                "id": data["id"],
                "num_round": len(dialogue),
                "dialogue": conversation,
            }
            logging.info(f"sample{data['id']} finished")
            ot.write(output_data)


if __name__ == "__main__":
    main()

# python multi_round.py
