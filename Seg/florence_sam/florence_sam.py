import supervision as sv
import torch
from seg_utils.video import generate_unique_name, create_directory, delete_directory
from seg_utils.florence import load_florence_model, run_florence_inference, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from seg_utils.sam import load_sam_image_model, run_sam_inference

DEVICE = torch.device("cuda")
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
# MASK_ANNOTATOR = sv.MaskAnnotator(
#     color=sv.Color.from_hex("#FFFFFF"),
#     color_lookup=sv.ColorLookup.INDEX
# )
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)


class Segmentation:

    def __init__(self):
        self.florence_model, self.florence_processor = load_florence_model(device=DEVICE)
        self.sam_image_model = load_sam_image_model(device=DEVICE)

    def annotate_image(self, image, detections):
        output_image = image.copy()
        output_image = MASK_ANNOTATOR.annotate(output_image, detections)
        output_image = BOX_ANNOTATOR.annotate(output_image, detections)
        # output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
        return output_image

    def process_image(self, image_input, prompt_text):
        texts = [prompt.strip() for prompt in prompt_text.split(",")]
        detections_list = []
        for text in texts:
            _, result = run_florence_inference(
                model=self.florence_model,
                processor=self.florence_processor,
                device=DEVICE,
                image=image_input,
                task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                text=text
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_input.size
            )
            detections = run_sam_inference(self.sam_image_model, image_input, detections)
            detections_list.append(detections)

        detections = sv.Detections.merge(detections_list)
        detections = run_sam_inference(self.sam_image_model, image_input, detections)
        mask = detections.mask
        return self.annotate_image(image_input, detections), mask
