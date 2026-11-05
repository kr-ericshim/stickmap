# adtof-stickmap-v2.pt

ADTOF Frame-RNN weights fine-tuned on drum transcriptions read from YouTube videos.

The original model and weights come from ADTOF by Mickaël Zehren, Marco Alunno and Paolo Bientinesi
(https://github.com/MZehren/ADTOF), loaded through ADTOF-pytorch (https://github.com/xavriley/ADTOF-pytorch).
ADTOF is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License, so these fine-tuned weights are released under the same license:
https://creativecommons.org/licenses/by-nc-sa/4.0/

Changes: the Frame-RNN was trained further on 131 songs of harvested ground truth. See tools/finetune.
