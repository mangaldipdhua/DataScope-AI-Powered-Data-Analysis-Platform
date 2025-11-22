import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Phi3ChatBot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.loaded = False
        self.init_model()
    
    def init_model(self):
        """Initialize Phi-3 model with detailed logging"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                BitsAndBytesConfig = None
            
            model_name = os.getenv('PHI3_MODEL_NAME', 'microsoft/Phi-3-mini-128k-instruct')
            device_str = os.getenv('DEVICE', 'cpu').lower()
            cache_dir = os.getenv('MODEL_CACHE_DIR', './model_cache')
            
            logger.info("="*60)
            logger.info("Initializing Phi-3 Model")
            logger.info("="*60)
            logger.info(f"Model: {model_name}")
            logger.info(f"Cache Directory: {cache_dir}")
            
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                logger.info(f"✓ Created cache directory: {cache_dir}")
            
            torch.cuda.empty_cache()
            
            if torch.cuda.is_available() and device_str != 'cpu':
                self.device = torch.device('cuda')
                logger.info("✓ CUDA detected - using GPU acceleration")
            else:
                self.device = torch.device('cpu')
                logger.info("✓ Using CPU (slower but works)")
            
            logger.info(f"Downloading Phi-3 model from Hugging Face (first time only, ~15GB)...")
            logger.info("This may take 5-30 minutes depending on internet speed...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            logger.info("✓ Tokenizer loaded successfully")
            
            load_kwargs = {
                'cache_dir': cache_dir,
                'trust_remote_code': True,
                'attn_implementation': 'eager'
            }
            
            if torch.cuda.is_available():
                load_kwargs['torch_dtype'] = torch.float16
                load_kwargs['device_map'] = 'auto'
                if BitsAndBytesConfig is not None:
                    try:
                        bnb_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16,
                            bnb_8bit_use_double_quant=True,
                            bnb_8bit_quant_type='nf8'
                        )
                        load_kwargs['quantization_config'] = bnb_config
                        logger.info("✓ Using 8-bit quantization for GPU")
                    except Exception as quant_err:
                        logger.warning(f"Could not use 8-bit quantization: {quant_err}")
                else:
                    logger.info("✓ 8-bit quantization not available, using standard float16")
            else:
                load_kwargs['torch_dtype'] = torch.float32
                load_kwargs['device_map'] = 'cpu'
                logger.info("✓ Loading model on CPU with optimizations")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            logger.info("✓ Model loaded successfully")
            
            self.model.eval()
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            self.loaded = True
            logger.info("✓ Phi-3 Model Ready!")
            logger.info("="*60)
            
        except Exception as e:
            self.loaded = False
            logger.error(f"✗ Failed to initialize Phi-3 model: {type(e).__name__}: {e}")
            logger.warning("Phi-3 model disabled. Will fall back to Gemini API.")
            logger.info("="*60)
    
    def generate_response(self, user_message: str, max_tokens: int = 512) -> Optional[str]:
        """Generate response using Phi-3"""
        if not self.loaded or not self.model or not self.tokenizer:
            logger.warning("Phi-3 model not loaded. Cannot generate response.")
            return None
        
        try:
            import torch
            
            logger.info(f"Generating Phi-3 response for: {user_message[:100]}...")
            
            inputs = self.tokenizer(
                user_message,
                return_tensors='pt',
                max_length=1024,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 512),
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_beams=1
                )
            
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            response = response.replace(user_message, '').strip()
            
            torch.cuda.empty_cache()
            
            logger.info("✓ Phi-3 response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"✗ Phi-3 generation failed: {type(e).__name__}: {e}")
            return None


phi3_bot = Phi3ChatBot()


def generate_chat_response(message: str) -> Optional[str]:
    """Generate chat response using Phi-3"""
    if phi3_bot.loaded:
        response = phi3_bot.generate_response(message)
        if response:
            return response
    return None
