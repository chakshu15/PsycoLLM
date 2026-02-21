import os
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, HfFolder, snapshot_download


class PsycoLLMChat:
    def __init__(
        self,
        model_name: str = "MACLAB-HFUT/PsycoLLM",
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name
        self.model_path = model_path
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface"

        self.messages: List[Dict[str, str]] = []

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self._setup_auth(use_auth_token)
        self._load_model_and_tokenizer()

    def _setup_auth(self, use_auth_token: Optional[str]):
        if use_auth_token:
            login(token=use_auth_token)
            self.auth_token = use_auth_token
        elif os.getenv("HUGGINGFACE_TOKEN"):
            login(token=os.getenv("HUGGINGFACE_TOKEN"))
            self.auth_token = os.getenv("HUGGINGFACE_TOKEN")
        elif HfFolder.get_token():
            self.auth_token = HfFolder.get_token()
        else:
            self.logger.warning(
                "No HuggingFace token provided. Public models only."
            )
            self.auth_token = None

    def _load_model_and_tokenizer(self):
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.logger.info(f"Loading model from local path: {self.model_path}")
                model_path = self.model_path
            else:
                self.logger.info(f"Downloading model {self.model_name} from HuggingFace...")
                model_path = snapshot_download(
                    repo_id=self.model_name,
                    token=self.auth_token,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=self.auth_token if not self.model_path else None
            )

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=self.auth_token if not self.model_path else None
            )

            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:

        if system_prompt and not self.messages:
            self.messages.append({"role": "system", "content": system_prompt})

        self.messages.append({"role": "user", "content": prompt})

        try:
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )

            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )

            new_tokens = outputs[0][len(input_ids[0]):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            self.messages.append({"role": "assistant", "content": response})

            return {"response": response}

        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            return {"error": str(e)}

    def chat(self, system_prompt: Optional[str] = None):
        print("Welcome to PsycoLLM! Type 'quit' or 'exit' to end.")

        if system_prompt:
            print(f"\nSystem Prompt:\n{system_prompt}\n")

        while True:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("\nGoodbye! Take care.")
                break

            response = self.generate_response(
                user_input,
                system_prompt=system_prompt if not self.messages else None
            )

            if "error" in response:
                print(f"\nError: {response['error']}")
            else:
                print("\nAssistant:", response["response"])

    def clear_history(self):
        self.messages = []
        self.logger.info("Chat history cleared.")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        chat_bot = PsycoLLMChat(
            model_name="MACLAB-HFUT/PsycoLLM",
            cache_dir="./model_cache"
        )

        system_prompt = (
            "角色：你是一名优秀的心理咨询助手，具有丰富的咨询经验。"
            "你性格乐观开朗、热情待人；逻辑清晰、善于倾听，具有强烈的同理心。"
            "任务：认真倾听用户困扰，使用引导性问题帮助用户思考，"
            "使用积极语言，提供可行建议，并遵循心理咨询伦理。"
        )

        chat_bot.chat(system_prompt=system_prompt)

    except Exception as e:
        print(f"Error starting PsycoLLM: {str(e)}")


if __name__ == "__main__":
    main()