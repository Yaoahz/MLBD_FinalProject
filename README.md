# MLBD_FinalProject
The final project of MLBD in Imperial College London, 2025.

Based on work in Papp, Á., Porod, W. & Csaba, G. Nanoscale neural network using non-linear spin-wave interference. Nat Commun 12, 6422 (2021).https://doi.org/10.1038/s41467-021-26711-z.
This study investigates spin-wave–based computation as a candidate architecture for low-energy machine learning. We focus on the inverse design of magnetic geometries using a differentiable micromagnetic simulation framework and evaluate their performance on multi-class audio classification tasks. Spin-wave propagation is shaped through a Y-junction waveguide network, and the approach is assessed in both three- and five-category classification settings.

The provided example script In3Out5.py defined a spin-wave propagation structure. The middle injection port inputs pre-processed audio data, and the rest of injection ports input two defined reference signals. This construction serve for five-category audio classification tasks.

The tool functions of audio pre-processing is in the Audio_preprocess.py

# Source of Audio data
The dataset used in this study is the Freesound Audio Tagging 2019 corpus, released on Kaggle and compiled by the Music Technology Group at Universitat Pompeu Fabra from Freesound content. Ground-truth annotations are provided at the clip level and indicate the presence of a sound category, thus serving as weak labels rather than frame-accurate transcriptions. The audio clips range in duration from approximately 0.3 to 30 s and are sampled at 44.1 kHz. All labels were assigned manually through the Freesound Annotator platform. Kaggle: Freesound Audio Tagging 2019. https://www.kaggle.com/competitions/freesound-audio-tagging-2019/overview.
