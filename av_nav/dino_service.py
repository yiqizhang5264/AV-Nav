"""Use explicit paths instead of the upstream script's hard-coded defaults."""
import argparse
import torch
import torchvision.transforms.functional as F
from groundingdino.util.inference import predict
from vlfm.vlm.grounding_dino import GroundingDINO
from vlfm.vlm.detections import ObjectDetections
from vlfm.vlm.server_wrapper import ServerMixin, host_model, str_to_image


class Service(ServerMixin, GroundingDINO):
    def __init__(self, device='cpu', **kwargs):
        self.inference_device=device
        super().__init__(device=torch.device(device),**kwargs)

    def predict(self,image,caption=None):
        # Match upstream preprocessing/filtering while passing the actual device
        # to GroundingDINO's inference utility (its default is otherwise CUDA).
        caption=self.caption if caption is None else caption
        tensor=F.normalize(F.to_tensor(image),mean=[.485,.456,.406],std=[.229,.224,.225])
        with torch.inference_mode():
            boxes,logits,phrases=predict(model=self.model,image=tensor,caption=caption,
                                        box_threshold=self.box_threshold,text_threshold=self.text_threshold,
                                        device=self.inference_device)
        detections=ObjectDetections(boxes,logits,phrases,image_source=image)
        detections.filter_by_class(caption[:-len(' .')].split(' . '))
        return detections

    def process_payload(self,payload):
        return self.predict(str_to_image(payload['image']),caption=payload['caption']).to_json()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    torch.set_num_threads(4)
    host_model(Service(config_path=args.config,weights_path=args.weights,device=args.device),name='gdino',port=args.port)
