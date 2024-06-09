class TemplateProcessor:
    def __init__(self, config):
        self.prompt_template = config.get("prompt_template", None)
        self.postprocessing_config = config.get("postprocessing_config", None)
        self.reference_template = config.get("reference_template", None)
    
    def create_prompt(self, data):
        """Creates a prompt using the loaded template and provided data."""
        try:
            prompt = self.prompt_template.format(**data)
        except KeyError as e:
            raise KeyError(f"Missing key in dataset for template: {e}")
        except IndexError as e:
            raise IndexError(f"Index error in template formatting: {e}")
        return prompt
    
    # reference: https://github.com/ywen666/gift4code/blob/db9100c7fd95ce5306179eb2292005e2bc6fba1b/eval/bigcode_eval/tasks/humaneval.py
    def format_humaneval(self, prompt, text):
        """Collates the model output for the humaneval format."""
        stop_sequences=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"]
        min_stop_index = len(text)
        for seq in stop_sequences:
            stop_index = text.find(seq)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return prompt + "\n" + text[:min_stop_index]

    def postprocess(self, generated_text_list, record):
        """Postprocess generated text."""
        if self.postprocessing_config is None: # no postprocessing
            return generated_text_list 
        
        postprocessed_text_list = []
        for generated_text in generated_text_list:
            postprocessed_text = generated_text
            for process in self.postprocessing_config['process']:
                if process == "humaneval":
                    postprocessed_text = self.format_humaneval(record["prompt"], postprocessed_text)
                elif process == "extract":
                    postprocessed_text = postprocessed_text #TODO: implement extract
                else:
                    raise ValueError(f"Unsupported postprocessing process: {process}")
                postprocessed_text_list.append(postprocessed_text)
        return postprocessed_text_list
    
    def create_reference(self, data):
        """Creates a reference using the loaded template and provided data."""
        try:
            reference = self.reference_template.format(**data)
        except KeyError as e:
            raise KeyError(f"Missing key in dataset for template: {e}")
        except IndexError as e:
            raise IndexError(f"Index error in template formatting: {e}")
        return reference
        
# TODO: refactoring
def load_template(config):
    """Load template from config."""
    Template = TemplateProcessor(config)
    return Template
