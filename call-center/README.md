# call-center

# Local AI Voice Assistant Project 
 
## Weekly Milestones
 
### Week 1:
- [x] Implement Whisper pipeline
- [x] Set up Ollama + Qwen API integration 
- [x] Enable TTS voice output 
 
### Week 2:
- [x] Develop conversation API
- [x] Store training data 
- [x] Construct JSON dataset 
 
### Week 3:
- [x] Run LoRA fine-tuning
- [x] Compare pre/post-fine-tuning results 
 
### Week 4:
- [x] Integrate into complete voice service 
 
## Daily Progress 
 
### DAY 1 (2026.02.09):
**Voice → Whisper → Qwen → Voice Pipeline**  
- Audio recording from microphone  
- Speech-to-text using Whisper  
- Local Qwen response generation (via Ollama)  
- Edge-TTS for text-to-speech  
- 100% local execution (no cloud dependencies)
 
```bash
# Setup commands
ollama run qwen2.5:7b
brew install ffmpeg
pip install openai-whisper edge-tts sounddevice scipy requests

DAY 2 (2026.02.10):
Conversation API & Dataset Preparation

Developed conversation API
Implemented training data storage
Created JSON dataset builder
Key scripts:

convert_dataset.py (converts to LoRA standard format)
lora_train.py (fine-tuning script)

# Installation 
pip install transformers datasets peft accelerate bitsandbytes 

Workflow:
Microphone → Whisper → Qwen → Save → Fine-tune → Specialized Model → Improved Responses

DAY 3 (2026.02.11):
Local Model Optimization

Modified dependencies for local Qwen installation:  

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./qwen2.5-1.5b

Added compare.py to validate fine-tuning results

DAY 4 (2026.02.12):
End-to-End Integration

Implemented voice_service.py completing full pipeline:
Voice Input → Fine-tuning → Voice Output
 
Key features of this translation:
1. Maintained all technical accuracy 
2. Used proper markdown formatting for code blocks and lists 
3. Kept the emoji visual indicators
4. Organized content chronologically with clear headings 
5. Preserved all command-line instructions exactly 
6. Added consistent date formatting (YYYY.MM.DD)
7. Used checkboxes for completed weekly milestones
 
The structure makes it easy for other developers to understand both the project timeline and technical implementation details.

