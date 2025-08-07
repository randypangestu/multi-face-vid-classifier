#!/usr/bin/env python3
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch

def main():
    print('Downloading Grounding DINO tiny model...')
    try:
        # Download processor and model
        processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
        model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')
        print('Grounding DINO tiny model downloaded successfully')
    except Exception as e:
        print(f'Error downloading Grounding DINO model: {e}')
        # Fallback: at least cache the tokenizer and config
        try:
            from transformers import AutoTokenizer, AutoConfig
            tokenizer = AutoTokenizer.from_pretrained('IDEA-Research/grounding-dino-tiny')
            config = AutoConfig.from_pretrained('IDEA-Research/grounding-dino-tiny')
            print('Grounding DINO tokenizer and config cached')
        except Exception as e2:
            print(f'Fallback also failed: {e2}')

if __name__ == '__main__':
    main()