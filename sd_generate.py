"""
Generate training images with a pretrained Stable Diffusion model (sd-turbo).

Makes three kinds of images for the cat / dog / neither classifier:
    cat      varied cats (breeds from your dataset, places, angles, light)
    dog      varied dogs (breeds from your dataset, places, angles, light)
    neither  people, vehicles, objects, scenes and OTHER animals (horse, rabbit, bird...)

Output:
    dataset_sd/train/cat/*.jpg
    dataset_sd/train/dog/*.jpg
    dataset_sd/train/neither/*.jpg
    dataset_sd/prompts.csv        (which prompt made which file - for checking)

Needs a GPU (Colab T4 is fine):
    pip install diffusers transformers accelerate

Usage:
    python sd_generate.py                         # 300 images per class
    python sd_generate.py --per-class 500
    python sd_generate.py --classes neither       # only one class
Re-running continues where it stopped (existing files are kept).
"""

import argparse
import csv
import os
import random

import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stabilityai/sd-turbo"
OUT_DIR = "dataset_sd/train"
SIZE = 256  # saved size (model makes 512x512, we shrink to keep files small)

CAT_BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
    "tabby", "black", "orange", "calico", "white", "gray", "tuxedo", "kitten",
]
DOG_BREEDS = [
    "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle", "Boxer",
    "Chihuahua", "English Cocker Spaniel", "English Setter", "German Shorthaired Pointer",
    "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond", "Leonberger",
    "Miniature Pinscher", "Newfoundland", "Pomeranian", "Pug", "Saint Bernard", "Samoyed",
    "Scottish Terrier", "Shiba Inu", "Staffordshire Bull Terrier", "Wheaten Terrier",
    "Yorkshire Terrier", "Labrador Retriever", "Golden Retriever", "German Shepherd",
    "puppy", "mixed breed",
]
VIEWS = [  # {a} = "a" or "an", {noun} = the animal
    "close-up portrait of {a} {noun}", "full body photo of {a} {noun}",
    "side view photo of {a} {noun}", "photo of {a} {noun} from far away",
    "photo of {a} {noun} taken from above", "photo of {a} {noun} looking at the camera",
    "photo of {a} {noun} sleeping", "candid photo of {a} {noun}",
    "low angle photo of {a} {noun}", "slightly blurry phone photo of {a} {noun}",
]
PET_PLACES = [
    "sitting on a sofa", "outdoors in a garden", "on green grass", "in a park",
    "on a wooden floor indoors", "on a bed", "on a city street", "in the snow",
    "at the beach", "in a kitchen", "next to a window", "on a carpet",
    "in a car", "on a balcony", "in front of a plain wall", "under a tree",
]
LIGHTS = [
    "natural daylight", "dim indoor lighting", "golden hour light", "flash photo at night",
    "overcast day", "bright sunlight", "soft window light", "low light",
]
STYLES = [
    "realistic photo", "photo taken with a phone", "candid snapshot",
    "amateur photo", "high quality photo",
]

# 'neither' = clearly NOT a cat or dog. Wolves, foxes and big cats are left out on purpose
# because they are too close to dogs and cats.
NEITHER_SUBJECTS = [
    # people
    "a man standing on a street", "a woman smiling at the camera", "a child playing in a park",
    "two people talking at a cafe", "a crowd walking in a city", "a person riding a bicycle",
    "a man cooking in a kitchen", "an elderly man sitting on a bench", "a family at a dinner table",
    "a student studying at a desk",
    # vehicles
    "a red car parked on a street", "a motorcycle on a road", "a bus in a city",
    "a truck on a highway", "a bicycle leaning on a wall", "a boat on a lake",
    "an airplane in the sky", "a train at a station",
    # places and scenes
    "a house with a garden", "a city skyline", "a mountain landscape", "a beach with waves",
    "a forest path", "a busy market street", "an empty living room", "a bedroom interior",
    "a kitchen counter", "an office building", "a bridge over a river", "a desert landscape",
    # food
    "a plate of pasta", "a bowl of fruit on a table", "a pizza on a table",
    "a cup of coffee on a desk", "a birthday cake", "a bowl of rice and curry",
    # objects
    "a laptop on a desk", "a smartphone on a table", "a wooden chair", "a sofa in a room",
    "a pair of shoes on the floor", "a backpack on a bench", "a bookshelf full of books",
    "a potted plant on a windowsill", "a bouquet of flowers", "a bottle of water on a table",
    "a clock on a wall", "a guitar leaning on a wall", "a television in a living room",
    "a bag of groceries", "a toolbox on a workbench", "a football on grass",
    # other animals (hard negatives)
    "a horse in a field", "a rabbit on grass", "a bird on a branch", "a cow in a meadow",
    "a sheep on a hill", "a hamster in a cage", "a goldfish in an aquarium",
    "a duck on a pond", "a parrot on a perch", "a deer in the woods",
    "an elephant walking", "a squirrel on a tree", "a turtle on a rock",
    "a pig on a farm", "a goat on a mountain", "a chicken in a yard", "a monkey on a tree",
]


def make_prompt(rng, kind):
    light = rng.choice(LIGHTS)
    style = rng.choice(STYLES)
    if kind == "neither":
        return f"{rng.choice(NEITHER_SUBJECTS)}, {light}, {style}"
    breed = rng.choice(CAT_BREEDS if kind == "cat" else DOG_BREEDS)
    animal = "cat" if kind == "cat" else "dog"
    if breed in ("kitten", "puppy"):
        noun = breed
    elif breed == "mixed breed":
        noun = "mixed breed dog"
    else:
        noun = f"{breed} {animal}"
    article = "an" if noun[0].lower() in "aeiou" else "a"
    subject = rng.choice(VIEWS).format(a=article, noun=noun)
    return f"{subject}, {rng.choice(PET_PLACES)}, {light}, {style}"


def generate(pipe, kind, n, batch, steps, seed, log):
    out = os.path.join(OUT_DIR, kind)
    os.makedirs(out, exist_ok=True)
    existing = len([f for f in os.listdir(out) if f.endswith(".jpg")])
    todo = n - existing
    if todo <= 0:
        print(f"{kind}: already has {existing} images, nothing to do")
        return
    offset = {"cat": 0, "dog": 1, "neither": 2}[kind]
    rng = random.Random(seed + offset * 1000 + existing)  # new prompts when resuming
    made = 0
    while made < todo:
        k = min(batch, todo - made)
        prompts = [make_prompt(rng, kind) for _ in range(k)]
        gens = [torch.Generator("cuda").manual_seed(rng.randrange(2 ** 31)) for _ in range(k)]
        images = pipe(prompt=prompts, num_inference_steps=steps,
                      guidance_scale=0.0, generator=gens).images
        for p, im in zip(prompts, images):
            name = f"sd_{kind}_{existing + made:04d}.jpg"
            im.convert("RGB").resize((SIZE, SIZE)).save(os.path.join(out, name), quality=92)
            log.writerow([name, p])
            made += 1
        if made % 20 < batch or made == todo:
            print(f"{kind}: {existing + made}/{n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--classes", nargs="+", default=["cat", "dog", "neither"],
                        choices=["cat", "dog", "neither"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU")

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)

    os.makedirs("dataset_sd", exist_ok=True)
    with open("dataset_sd/prompts.csv", "a", newline="") as f:
        log = csv.writer(f)
        for kind in args.classes:
            generate(pipe, kind, args.per_class, args.batch, args.steps, args.seed, log)
            f.flush()
    print("\nDone. Look at the images before training (see the preview step).")


if __name__ == "__main__":
    main()
