from transformers import pipeline

class Model(object):
    def __init__(self, config):
        self.model_path = config.get("model_path", None)
        self.tokenizer_path = config.get("tokenizer_path", self.model_path)
        self.api_key = config.get("api_key", None)
        self.generation_config = config.get("generation_config", {})
    

class HuggingFaceModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.generator = pipeline("text-generation", model=self.model_path, tokenizer=self.tokenizer_path,
                                device_map="auto", use_auth_token=self.api_key)

    def generate(self, prompt):
        if self.generation_config is {}: # default generation config
            default_generation_config = {
                "max_new_tokens": 256, 
                "do_sample": False, 
                "repetition_penalty": 1.0, 
                "num_return_sequences": 1, 
                "return_full_text": False
            }

        generated_texts = self.generator(prompt, **self.generation_config)
        return [text['generated_text'] for text in generated_texts]
    

def load_model(config):
    """Load model from config."""
    model_type = config.get('model_type', None)
    if model_type == "huggingface" or model_type is None:
        model = HuggingFaceModel(config)
    elif model_type == "openai":
        model = OpenAIModel(config) #TODO:OpenAIModel
    elif model == "bedrock":
        model = BedrockModel(config) #TODO:BedrockModel
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

