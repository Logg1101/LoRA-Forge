import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


def generate_buckets(target_res=1024, min_res=256, max_res=2048, step=64):
    """Generates standard VAE-compatible aspect ratio buckets within resolution bounds."""
    target_pixels = target_res * target_res
    ratios = [1.0, 4/3, 3/4, 16/9, 9/16, 2/3, 3/2, 1/2, 2/1, 5/4, 4/5, 7/9, 9/7]
    buckets = []

    for r in ratios:
        h = math.sqrt(target_pixels / r)
        w = h * r
        h = int(round(h / float(step)) * step)
        w = int(round(w / float(step)) * step)

        h = max(min_res, min(max_res, h))
        w = max(min_res, min(max_res, w))

        if (w, h) not in buckets:
            buckets.append((w, h))
    return buckets


def get_closest_bucket(w, h, buckets):
    """Finds the bucket that most closely matches the image's original aspect ratio."""
    aspect = w / h
    best_bucket = min(buckets, key=lambda b: abs((b[0] / b[1]) - aspect))
    return best_bucket


class ARBSampler(Sampler):
    """Groups images of the same resolution bucket into batches."""

    def __init__(self, target_shapes, batch_size, drop_last=False):
        self.batch_size = max(1, batch_size)
        self.drop_last = drop_last
        self.buckets = {}

        for idx, shape in enumerate(target_shapes):
            if shape not in self.buckets:
                self.buckets[shape] = []
            self.buckets[shape].append(idx)

    def __iter__(self):
        batches = []
        for shape, indices in self.buckets.items():
            if not indices:
                continue
            shuffled_indices = list(indices)
            random.shuffle(shuffled_indices)

            # If drop_last is False and we have remainder, pad with replacement from same bucket
            remainder = len(shuffled_indices) % self.batch_size
            if not self.drop_last and remainder != 0:
                pad_count = self.batch_size - remainder
                shuffled_indices.extend(random.choices(indices, k=pad_count))

            for i in range(0, len(shuffled_indices), self.batch_size):
                batch = shuffled_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)

        # Fallback safeguard: if drop_last was True but yielded 0 batches across all buckets,
        # create at least one padded batch so any non-empty dataset produces at least 1 batch.
        if len(batches) == 0 and any(len(idx_list) > 0 for idx_list in self.buckets.values()):
            for shape, indices in self.buckets.items():
                if indices:
                    pad_count = (self.batch_size - (len(indices) % self.batch_size)) % self.batch_size
                    padded = list(indices) + random.choices(indices, k=pad_count)
                    for i in range(0, len(padded), self.batch_size):
                        batches.append(padded[i:i + self.batch_size])
                    break

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if not self.drop_last:
            return sum(math.ceil(len(indices) / self.batch_size) for indices in self.buckets.values() if len(indices) > 0)
        else:
            base_count = sum(len(indices) // self.batch_size for indices in self.buckets.values())
            if base_count == 0 and any(len(indices) > 0 for indices in self.buckets.values()):
                return 1
            return base_count


class ARBDataset(Dataset):
    """
    Aspect-Ratio-Bucketed dataset supporting repeats, tag shuffling, keep tokens,
    caption dropout, data augmentations (flip/color/crop), and latent RAM caching.
    """

    def __init__(
        self,
        dataset_dir,
        max_resolution=1024,
        repeats=1,
        caption_extension=".txt",
        shuffle_captions=False,
        keep_tokens=0,
        caption_dropout_rate=0.0,
        tag_dropout_rate=0.0,
        horizontal_flip=False,
        color_augmentation=False,
        random_crop=False,
        min_bucket_res=256,
        max_bucket_res=2048,
        bucket_step=64,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.repeats = repeats
        self.caption_ext = caption_extension
        self.shuffle_captions = shuffle_captions
        self.keep_tokens = keep_tokens
        self.caption_dropout_rate = caption_dropout_rate
        self.tag_dropout_rate = tag_dropout_rate
        self.horizontal_flip = horizontal_flip
        self.color_augmentation = color_augmentation
        self.random_crop = random_crop

        self.buckets = generate_buckets(
            target_res=max_resolution,
            min_res=min_bucket_res,
            max_res=max_bucket_res,
            step=bucket_step,
        )
        self.is_cached = False

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        self.raw_image_paths = []
        self.raw_captions = []
        self.raw_target_shapes = []

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        print("🔍 Scanning dataset and calculating ARB buckets...")
        for file in sorted(self.dataset_dir.rglob("*")):
            if file.suffix.lower() in valid_extensions:
                try:
                    with Image.open(file) as img:
                        w, h = img.size
                    target_shape = get_closest_bucket(w, h, self.buckets)
                    self.raw_image_paths.append(file)
                    self.raw_captions.append(self._find_caption(file, self.caption_ext))
                    self.raw_target_shapes.append(target_shape)
                except Exception as e:
                    print(f"⚠️ Warning: Skipping unreadable image {file} - {e}")

        if len(self.raw_image_paths) == 0:
            raise ValueError(f"No valid images found in {self.dataset_dir}.")

        # Expand by repeats
        self.image_paths = self.raw_image_paths * self.repeats
        self.captions = self.raw_captions * self.repeats
        self.target_shapes = self.raw_target_shapes * self.repeats

        # Print bucket distribution
        bucket_counts = {}
        for shape in self.raw_target_shapes:
            bucket_counts[shape] = bucket_counts.get(shape, 0) + 1
        print(f"📊 Found {len(self.raw_image_paths)} unique images (× {self.repeats} repeats = {len(self.image_paths)} samples) across {len(bucket_counts)} buckets:")
        for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
            print(f"   {bucket[0]}x{bucket[1]}: {count} images")

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _find_caption(image_path, extension=".txt"):
        caption_path = image_path.with_suffix(extension)
        if caption_path.exists():
            with open(caption_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        # Fallback to .txt or .caption if custom ext not found
        for ext in [".txt", ".caption"]:
            alt = image_path.with_suffix(ext)
            if alt.exists():
                with open(alt, "r", encoding="utf-8") as f:
                    return f.read().strip()
        return ""

    def _process_caption(self, caption: str) -> str:
        """Apply caption dropout, tag dropout, tag shuffling, and keep-tokens."""
        if not caption:
            return ""

        # Whole caption dropout
        if self.caption_dropout_rate > 0 and random.random() < self.caption_dropout_rate:
            return ""

        # Tag-level processing (comma-separated tags)
        if "," in caption:
            tags = [t.strip() for t in caption.split(",") if t.strip()]
            if not tags:
                return ""

            fixed = tags[:self.keep_tokens]
            flexible = tags[self.keep_tokens:]

            # Individual tag dropout (protecting keep_tokens)
            if self.tag_dropout_rate > 0 and flexible:
                flexible = [t for t in flexible if random.random() >= self.tag_dropout_rate]

            # Tag shuffling
            if self.shuffle_captions and flexible:
                random.shuffle(flexible)

            combined = fixed + flexible
            return ", ".join(combined)

        return caption

    def _load_and_preprocess(self, index):
        """Load image, apply augmentations, and resize/crop to bucket resolution."""
        image_path = self.image_paths[index]
        target_w, target_h = self.target_shapes[index]

        try:
            image = Image.open(image_path).convert("RGB")

            # Augmentation: Horizontal Flip
            if self.horizontal_flip and random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Augmentation: Color Jitter
            if self.color_augmentation and random.random() < 0.3:
                enh_b = ImageEnhance.Brightness(image)
                image = enh_b.enhance(random.uniform(0.9, 1.1))
                enh_c = ImageEnhance.Color(image)
                image = enh_c.enhance(random.uniform(0.9, 1.1))

            orig_w, orig_h = image.size
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Crop
            if self.random_crop:
                max_x = max(0, new_w - target_w)
                max_y = max(0, new_h - target_h)
                left = random.randint(0, max_x) if max_x > 0 else 0
                top = random.randint(0, max_y) if max_y > 0 else 0
            else:
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2

            image = image.crop((left, top, left + target_w, top + target_h))

            pixel_values = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0)
            pixel_values = pixel_values.permute(2, 0, 1)
            pixel_values = (pixel_values - 0.5) / 0.5

        except (UnidentifiedImageError, OSError):
            pixel_values = torch.zeros((3, target_h, target_w))

        return pixel_values

    def __getitem__(self, index):
        caption = self._process_caption(self.captions[index])
        if self.is_cached:
            item = {
                "latents": self.cached_latents[index],
                "captions": caption,
            }
            if self.cached_prompt_embeds is not None:
                item["prompt_embeds"] = self.cached_prompt_embeds[index]
            if self.cached_pooled_embeds is not None:
                item["pooled_prompt_embeds"] = self.cached_pooled_embeds[index]
            return item

        pixel_values = self._load_and_preprocess(index)

        return {
            "pixel_values": pixel_values,
            "captions": caption,
        }

    @torch.no_grad()
    def cache_latents_and_embeds(
        self, vae, tokenizer, text_encoder, device, dtype,
        model_type="SDXL", tokenizer_2=None, text_encoder_2=None,
    ):
        """Pre-encode all images and optionally captions to host RAM."""
        is_sdxl = model_type == "SDXL"
        is_flux = "Flux" in model_type
        is_dual = is_sdxl or is_flux

        print("⚡ Caching latents and text embeddings to RAM...")

        vae.to(device)
        if text_encoder is not None:
            text_encoder.to(device)
        if is_dual and text_encoder_2 is not None:
            text_encoder_2.to(device)

        self.cached_latents = []
        has_dynamic_captions = self.shuffle_captions or self.caption_dropout_rate > 0 or self.tag_dropout_rate > 0
        should_cache_text = (text_encoder is not None and tokenizer is not None and not has_dynamic_captions)
        if has_dynamic_captions and (text_encoder is not None and tokenizer is not None):
            print("   ℹ️ Dynamic captioning (shuffling/dropout) active — image latents cached; text embeddings computed dynamically.")
        self.cached_prompt_embeds = [] if should_cache_text else None
        self.cached_pooled_embeds = [] if (should_cache_text and is_dual) else None

        for i in range(len(self)):
            # VAE Encode
            pixel_values = self._load_and_preprocess(i).unsqueeze(0).to(device, dtype=vae.dtype)
            latent = vae.encode(pixel_values).latent_dist.sample()

            if is_flux:
                shift = getattr(vae.config, "shift_factor", 0.0)
                scale = getattr(vae.config, "scaling_factor", 1.0)
                latent = (latent - shift) * scale
            else:
                scale = getattr(vae.config, "scaling_factor", 1.0)
                latent = latent * scale

            self.cached_latents.append(latent.squeeze(0).cpu())

            # Text Encode (only if text encoders are provided for caching)
            if should_cache_text:
                caption = self._process_caption(self.captions[i])

                if is_dual:
                    inp_1 = tokenizer(
                        caption, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt",
                    ).input_ids.to(device)

                    inp_2 = tokenizer_2(
                        caption, padding="max_length",
                        max_length=tokenizer_2.model_max_length,
                        truncation=True, return_tensors="pt",
                    ).input_ids.to(device)

                    out_1 = text_encoder(inp_1, output_hidden_states=True)
                    out_2 = text_encoder_2(inp_2, output_hidden_states=True)

                    if is_flux:
                        pooled = out_1.pooler_output
                        embeds = out_2.last_hidden_state
                    else:
                        embeds = torch.cat(
                            [out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1
                        )
                        pooled = out_2.text_embeds

                    self.cached_prompt_embeds.append(embeds.squeeze(0).cpu())
                    self.cached_pooled_embeds.append(pooled.squeeze(0).cpu())

                else:
                    inputs = tokenizer(
                        caption, padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt",
                    )
                    embeds = text_encoder(inputs.input_ids.to(device))[0]
                    self.cached_prompt_embeds.append(embeds.squeeze(0).cpu())

            if (i + 1) % 50 == 0 or i == len(self) - 1:
                print(f"   Cached {i + 1}/{len(self)} samples")

        self.is_cached = True
        print(f"✅ Caching complete. {len(self)} samples ready in RAM.")

    @staticmethod
    def validate_dataset_dir(dataset_dir: str) -> list[str]:
        """Scans dataset directory and returns a list of actionable warnings."""
        warnings = []
        p = Path(dataset_dir)
        if not p.exists() or not p.is_dir():
            return [f"Directory does not exist: {dataset_dir}"]

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        images = [f for f in p.rglob("*") if f.suffix.lower() in valid_exts]

        if not images:
            return [f"No image files found in {dataset_dir} (.jpg, .png, .webp)"]

        missing_captions = 0
        corrupt_images = 0
        extreme_res = 0

        for img_path in images:
            txt_path = img_path.with_suffix(".txt")
            cap_path = img_path.with_suffix(".caption")
            if not txt_path.exists() and not cap_path.exists():
                missing_captions += 1

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w < 128 or h < 128:
                        extreme_res += 1
            except Exception:
                corrupt_images += 1

        if missing_captions > 0:
            warnings.append(f"{missing_captions} of {len(images)} images are missing accompanying .txt caption files.")
        if corrupt_images > 0:
            warnings.append(f"{corrupt_images} images could not be read (corrupted or unreadable).")
        if extreme_res > 0:
            warnings.append(f"{extreme_res} images are smaller than 128x128 resolution.")

        return warnings


def create_dataloader(
    dataset_dir,
    batch_size=1,
    max_resolution=1024,
    repeats=1,
    caption_extension=".txt",
    shuffle_captions=False,
    keep_tokens=0,
    caption_dropout_rate=0.0,
    tag_dropout_rate=0.0,
    horizontal_flip=False,
    color_augmentation=False,
    random_crop=False,
    min_bucket_res=256,
    max_bucket_res=2048,
    bucket_step=64,
    num_workers=2,
    pin_memory=True,
):
    """Factory function creating the ARB Dataset and DataLoader."""
    dataset = ARBDataset(
        dataset_dir=dataset_dir,
        max_resolution=max_resolution,
        repeats=repeats,
        caption_extension=caption_extension,
        shuffle_captions=shuffle_captions,
        keep_tokens=keep_tokens,
        caption_dropout_rate=caption_dropout_rate,
        tag_dropout_rate=tag_dropout_rate,
        horizontal_flip=horizontal_flip,
        color_augmentation=color_augmentation,
        random_crop=random_crop,
        min_bucket_res=min_bucket_res,
        max_bucket_res=max_bucket_res,
        bucket_step=bucket_step,
    )
    sampler = ARBSampler(dataset.target_shapes, batch_size)

    extra_kwargs = {}
    if num_workers > 0:
        extra_kwargs["persistent_workers"] = True
        extra_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **extra_kwargs,
    )

    return dataset, dataloader


def get_dataset_diagnostics(
    dataset: ARBDataset,
    batch_size: int = 1,
    grad_accum_steps: int = 1,
    epochs: int = 1,
    max_train_steps: int = 0,
) -> dict:
    """
    Computes exact training diagnostics and step counts before training begins.
    """
    unique_images = len(dataset.raw_image_paths)
    effective_images = len(dataset.image_paths)
    
    # Calculate images per bucket
    bucket_counts = {}
    for shape in dataset.raw_target_shapes:
        bucket_counts[f"{shape[0]}x{shape[1]}"] = bucket_counts.get(f"{shape[0]}x{shape[1]}", 0) + 1
        
    sampler = ARBSampler(dataset.target_shapes, batch_size)
    batches_per_epoch = len(sampler)
    
    # Optimizer steps per epoch accounting for gradient accumulation
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / max(1, grad_accum_steps))
    total_expected_optimizer_steps = optimizer_steps_per_epoch * epochs
    if max_train_steps > 0:
        total_expected_optimizer_steps = min(total_expected_optimizer_steps, max_train_steps)
        
    return {
        "unique_images": unique_images,
        "repeats": dataset.repeats,
        "effective_images": effective_images,
        "buckets_count": len(bucket_counts),
        "bucket_distribution": bucket_counts,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "epochs": epochs,
        "max_train_steps": max_train_steps,
        "total_expected_optimizer_steps": total_expected_optimizer_steps,
    }