from typing import List, Tuple
import os

import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from arg_utils import Args
from data import get_loaders


# ColPali imports (requires colpali repo cloned and installed)
from colpali_engine.interpretability import get_similarity_maps_from_embeddings
from colpali_engine.interpretability.similarity_maps import get_similarity_map
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device


# -------------------------------
# Utility functions
# -------------------------------

def prepare_similarity_map(
    image_size: Tuple[int, int],
    similarity_maps: torch.Tensor,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Return similarity map image and array for given embeddings.
    """
    img, arr = get_similarity_map(
        image_size=image_size,
        similarity_map=similarity_maps[0],
    )
    return img, arr


def process_loader(
    loader,
    model,
    processor,
    device,
    saving_path: str,
):
    """
    Process a dataloader to generate and save similarity maps.
    """
    for batch in tqdm(loader, desc="Processing batches"):
        batch_images = batch["images"].to(device)
        batch_queries = batch["queries"].to(device)

        # Forward passes
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            query_embeddings = model.forward(**batch_queries)

        # Compute similarity maps
        n_patches = processor.get_n_patches(
            image_size=(0, 0),  # TODO: replace with real size if needed
            patch_size=model.patch_size,
        )
        image_mask = processor.get_image_mask(batch_images)

        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )

        ids = batch["question_ids"]
        sizes = batch["sizes"]

        for idx, size, similarity_maps in zip(ids, sizes, batched_similarity_maps):
            # Average across queries
            similarity_maps = similarity_maps.mean(0).unsqueeze(0)

            # Get similarity map
            _, maps_arr = prepare_similarity_map(size, similarity_maps)

            # Resize and save
            resized_maps_arr = cv2.resize(maps_arr, (size[0], size[1]))
            out_path = os.path.join(saving_path, f"{idx}.png")
            cv2.imwrite(out_path, resized_maps_arr * 255)


# -------------------------------
# Main entry point
# -------------------------------

def main():
    args = Args().parse_args(known_only=True)

    # Paths
    saving_path = args.colpali_path
    os.makedirs(saving_path, exist_ok=True)

    # Device
    device = get_torch_device("auto")

    # Model + Processor
    model_name = "vidore/colpali-v1.2"
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)

    # Dataloaders
    train_loader, val_loader, test_loader = get_loaders(
        processor, args.data_path, batch_size=args.batch_size
    )

    # Process splits
    print("Processing test set")
    process_loader(test_loader, model, processor, device, saving_path)

    print("Processing train set")
    process_loader(train_loader, model, processor, device, saving_path)

    print("Processing validation set")
    process_loader(val_loader, model, processor, device, saving_path)


if __name__ == "__main__":
    main()
