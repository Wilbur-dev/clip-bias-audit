import torch
import open_clip


class CLIPEncoder:
    def __init__(self, model_name = "ViT-B-32", pretrained = "openai", device = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, images):
        images = images.to(self.device)
        emb = self.model.encode_image(images)
        emb = emb / emb.norm(dim = -1, keepdim = True)
        return emb.cpu()

    @torch.no_grad()
    def encode_texts(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim = -1, keepdim = True)
        return emb.cpu()