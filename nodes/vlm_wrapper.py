from time import perf_counter
from loguru import logger
from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info


class VLMWrapper:
    """Wrapper around VLM class
    """
    model_path:str = r"D:\models\Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"
    min_pixels:int = 256*28*28
    max_pixels:int = 1024*28*28

    def __init__(self, model_path: str | None = None,
                 min_pixels: int | None = None,
                 max_pixels: int | None = None,
                 max_new_tokens: int = 512):
        if model_path:
            self.model_path = model_path

        if min_pixels:
            self.min_pixels = min_pixels

        if max_pixels:
            self.max_pixels = max_pixels

        self.max_new_tokens = max_new_tokens

        logger.info("Attempting to load VLM")
        t0 = perf_counter()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        logger.info("Tokenizer successfully loaded")
        self.processor = AutoProcessor.from_pretrained(self.model_path,
                                            min_pixels=self.min_pixels,
                                            max_pixels=self.max_pixels
                                            )
        logger.info("Processor successfully loaded")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype = torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config
        )

        t0 = perf_counter() - t0
        logger.info(f"Model successfully loaded in {t0:.2f}s")


    def __call__(self, text: str, image: str | Image.Image) -> list[str]:
        """Query VLM with multimodal prompt

        Args:
            text (str): text prompt
            image (str | Image.Image): path/to/image.png or PIL.Image

        Returns:
            list[str]: VLM result
        """
        t0 = perf_counter()
        with torch.no_grad():
            logger.info("Prepare inputs")

            if not isinstance(image, Image.Image):
                pil_image = Image.open(image)
            else:
                pil_image = image

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image", "image": pil_image}
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # Generate
            logger.info("Generating")
            try:
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    temperature=0.7,
                )
                
                t0 = perf_counter() - t0
                logger.info(f"Success in {t0:.2f}s")
            except Exception as e:
                logger.exception(f"Error during model inference: {str(e)}")

            return result


if __name__ == "__main__":
    vlm = VLMWrapper()
    # Test inference
    # text = "Enemy attack rate is depicted with yellow sword icon, and its defense rate is depicted with yellow shield icon. What are the values of these rates?"
    # result = vlm(text="Enemy Arts attack rate is depicted with blue icon, and its Arts defense rate is depicted with blue shield icon. What are the values of these rates?",
    #     image_path="screenshot.png")
    
    result = vlm(text="List numbers that are located below a thick blue line.",
        image_path=r"imgs\action_table.png")

    print(result[0])
    