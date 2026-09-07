"""Use explicit paths instead of the upstream script's hard-coded defaults."""
import argparse
from vlfm.vlm.grounding_dino import GroundingDINO
from vlfm.vlm.server_wrapper import ServerMixin, host_model, str_to_image


class Service(ServerMixin, GroundingDINO):
    def process_payload(self,payload):
        return self.predict(str_to_image(payload['image']),caption=payload['caption']).to_json()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--port',type=int,required=True)
    args=p.parse_args()
    host_model(Service(config_path=args.config,weights_path=args.weights),name='gdino',port=args.port)
