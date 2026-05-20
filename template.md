---
title: Gemini Omni Flash
description: A GitHub Pages model card template based on the Google DeepMind Gemini Omni Flash model card structure.
---

# Gemini Omni Flash

**Published:** May 2026  
**Model type:** Multimodal generative model

Model Cards are intended to provide developers with essential, summarized
information on models, including overviews of known limitations and mitigation
approaches. Model cards may be updated from time to time; for example, to
include updated evaluations as the model is improved or revised.

> Use this file as a GitHub Pages Markdown template. Replace the example
> Gemini Omni Flash content with details for your own model while keeping the
> model card structure below.

## Contents

1. [Model Information](#model-information)
2. [Model Data](#model-data)
3. [Implementation and Sustainability](#implementation-and-sustainability)
4. [Distribution](#distribution)
5. [Evaluation](#evaluation)
6. [Intended Usage and Limitations](#intended-usage-and-limitations)
7. [Ethics and Content Safety](#ethics-and-content-safety)

## Model Snapshot

| Field | Summary |
| --- | --- |
| Inputs | Text strings, images, audio, and video files. |
| Outputs | High-quality, high-resolution video with audio. |
| Architecture | Transformer-based model with native multimodal support. |

---

## Model Information

### Description

Gemini Omni Flash is our next step towards models that can create and edit
anything from any input, starting with video. It combines Gemini's intelligence
with generative media models, representing a leap forward in world
understanding, multimodality, and editing. Gemini Omni Flash enables
high-quality video creation and a more natural way to edit videos through
conversation.

### Inputs

Text strings, images, audio, and video files.

### Outputs

High-quality, high-resolution video with audio.

### Architecture

Gemini Omni Flash is a transformer-based model with native multimodal support
for text, vision, video, and audio inputs.

---

## Model Data

### Training Dataset

Gemini Omni Flash was trained on audio, video, image, and text data. Audio and
video datasets were annotated with text captions at different levels of detail.

### Training Data Processing

Training videos were filtered for compliance, safety, and quality metrics, then
deduplicated semantically.

---

## Implementation and Sustainability

### Hardware

Gemini Omni Flash was trained using Google's Tensor Processing Units (TPUs),
which are designed to handle the massive computations involved in training large
models. TPUs can speed up training compared to CPUs and often provide large
amounts of high-bandwidth memory for handling large models and batch sizes.

The efficiencies gained through the use of TPUs are aligned with Google's
commitment to operate sustainably.

### Software

Training was done using JAX and ML Pathways.

---

## Distribution

Gemini Omni Flash is distributed through documented channels.

> Replace this section with API, product, licensing, deployment, and access
> details for your own model.

---

## Evaluation

### Approach

Evaluations for T2VA, I2VA, R2VA, video editing, and image generation are
expected to be shared when the model rolls out to developers and enterprise
customers via APIs.

---

## Intended Usage and Limitations

### Benefit and Intended Usage

Gemini Omni Flash can be used to generate high-quality, high-resolution videos
from any input in a wide range of visual styles. The model is able to follow
simple and complex instructions, simulate real-world physics, and edit videos
through conversation.

### Known Limitations

Maintaining complete consistency throughout edits, generating scenes with
complex motion, and rendering perfectly accurate text remain challenging.

### Acceptable Usage

Google's Generative AI Prohibited Use Policy applies to uses of the model in
accordance with the applicable terms of service.

> Replace this section with the policies and restrictions that govern your own
> model.

---

## Ethics and Content Safety

### Evaluation Approach

Gemini Omni Flash was developed in partnership with internal safety, security,
and responsibility teams. A range of evaluations and red teaming activities were
conducted to help improve model safety and inform decision-making.

Evaluation types included but were not limited to:

- Training and development evaluations, including automated and human
  evaluations carried out continuously throughout and after training.
- Human red teaming conducted by specialist teams outside the model development
  team.
- Automated red teaming to dynamically evaluate safety and security
  considerations at scale.
- Ethics and safety reviews conducted ahead of release.

### Responsible Innovation

AI multimodal generation and editing tools can help lower barriers to entry and
transform education through personalized audio-visual content. Beyond direct
applications, high-quality synthetic data can help accelerate innovation in
robotics, computer vision, and generative 3D technologies.

Such advanced multimodal creation capabilities require a proactive approach to
safety. The main content safety areas are:

1. Intentional adversarial misuse of the model.
2. Unintentional model failure modes through benign use.

Safety mitigations included:

- **Pre-training mitigations:** Diverse synthetic captioning to improve the
  model's ability to accurately represent a wide variety of concepts.
- **Post-training mitigations:** Production filters and SynthID watermarking to
  help verify AI-generated content.

As part of editing videos, Gemini Omni Flash is capable of changing people's
speech. For now, this capability is restricted while additional safety and
responsibility work continues.

---

## Latest Model Cards

- Lyria 3
- Gemini 3.1 Pro
- Gemini 3.1 Flash Image
- Gemini 3.1 Flash-Lite
- Gemini 3.1 Flash Live
- Veo 3.1 Lite
