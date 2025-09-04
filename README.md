# MLBD_FinalProject
The final project of MLBD in Imperial College London, 2025.

Based on work in Papp, Á., Porod, W. & Csaba, G. Nanoscale neural network using non-linear spin-wave interference. Nat Commun 12, 6422 (2021).https://doi.org/10.1038/s41467-021-26711-z.
This project explored spinwave-based computation as a candidate architecture for low-energy machine learning. The project focus on the inverse design of magnetic geometries using a differentiable micromagnetic simulation framework and evaluate performance on multi-category audio classification. We shaped spin-wave propagation through a Y-junction waveguide network. In three and five-category audio classification tasks

The provided example script In3Out5.py defined a spin-wave propagation structure. The middle injection port inputs pre-processed audio data, and the rest of injection ports input two defined reference signals. This construction serve for five-category audio classification tasks.

# Source of Audio data
The dataset that we used is from ‘Freesound Audio Tagging 2019’ on Kaggle. It was compiled by the Music Technology Group at Universitat Pompeu Fabra using content from Freesound. Ground-truth labels are provided at the clip level and indicate the presence of a sound category in each clip, so they constitute weak labels or tags rather than frame-accurate annotations. The audio clips vary in duration from approximately 0.3 to 30 s and are sampled at 44.1 kHz. The audio content from FSD has been manually labeled by humans following a data labeling process using the Freesound Annotator platform. https://www.kaggle.com/competitions/freesound-audio-tagging-2019/overview
