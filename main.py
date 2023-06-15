from huggingface_hub import notebook_login, hf_hub_download
import glob
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
import os
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
from transformers import pipeline

def main():
    # login()
    # # downloadDaset()
    # label2id, id2label = getLabel()
    # loadModelFineTune(label2id, id2label)
    # #一度ここまで実行するのが良い
    # prepareDatasetsTraining()
    # trainModel()
    sample_test_video = evaluate_test()



def login():
    notebook_login()

def downloadDaset():
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"
    file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")

def getLabel():
    path = "/Users/jonmac/jon/研究/手話/test/dataset/train/*" #trainまでの絶対パス
    path_list=sorted(glob.glob(path))
    label2id = {label.replace("/Users/jonmac/jon/研究/手話/test/daset/train/", ""): i for i, label in enumerate(path_list)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label

def loadModelFineTune(label2id, id2label):
    global image_processor, model, model_ckpt
    model_ckpt = "MCG-NJU/videomae-base"
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

def prepareDatasetsTraining():
    global train_dataset, val_dataset, test_dataset
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

    train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ])
    dataset_root_path = "/Users/jonmac/jon/研究/手話/test/dataset" #subsetまでのパス
    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )
    val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ])   

    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    test_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

def trainModel():
    global args
    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-finetuned-ucf101-subset"
    num_epochs = 4
    batch_size = 10
    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    )

def evaluate_test():
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    def collate_fn(examples):
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    
    trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    )
    train_results = trainer.train()
    sample_test_video = next(iter(test_dataset))
    model_path = "./videomae-base-finetuned-ucf101-subset/checkpoint-10"
    video_cls = pipeline("video-classification", model=model_path)
    video_cls("https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/v_BasketballDunk_g14_c06.avi")
    print(video_cls)
    



if __name__ == "__main__":
    main()