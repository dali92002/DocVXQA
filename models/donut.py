
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, DonutProcessor, BartConfig
import torch


def init_donut(ckpt, lowreso=False):
	if lowreso:
		# Lower resolution for stable training
		max_length = 128
		image_size = [1280, 960]

		# during pre-training, a LARGER image size was used [2560, 1920]
		config = VisionEncoderDecoderConfig.from_pretrained(ckpt)
		config.encoder.image_size = image_size
		config.decoder.max_length = max_length
		model = VisionEncoderDecoderModel.from_pretrained(ckpt, config=config)

		# TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
		# https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602
		image_processor = DonutProcessor.from_pretrained(ckpt)
		image_processor.feature_extractor.size = image_size[::-1] # REVERSED (width, height)
		image_processor.feature_extractor.do_align_long_axis = False

		image_processor.tokenizer.add_tokens([
			"<s_docvqa>", "<yes/>", "<no/>", "<s_question>", "<s_answer>", "</s_answer>", "</s_question>"
		])
		model.decoder.resize_token_embeddings(len(image_processor.tokenizer))
	else:
		# Follow the exact model's config
		image_processor = DonutProcessor.from_pretrained(ckpt)
		model = VisionEncoderDecoderModel.from_pretrained(ckpt)
		
	return model, image_processor


def get_donut_model(task=None, ckpts=None):
	model_ckpt = "naver-clova-ix/donut-base-finetuned-docvqa"
	model, processor = init_donut(model_ckpt,
							   lowreso=(model_ckpt!='naver-clova-ix/donut-base-finetuned-docvqa'))
	
	if ckpts is not None:
		model.load_state_dict(torch.load(ckpts))
	return model, processor