# Scripts used to create figures:

In the top-level directory, each file with name matching `fig_X_*.py` is the script used to render figure X from the paper.

# Python implementation of Berstein & Spicer model:

The Python implementation of the Bernstein & Spicer model (['Line source representation for laser-generated ultrasound in aluminum', Bernstein & Spicer, JASA, 2000](https://doi.org/10.1121/1.428422)) is stored in the module [`bernstein_and_spicer_model.py`](https://github.com/MatthewRiding/Figures-laser-s-wave-DAS/blob/main/bernstein_and_spicer_model.py) in the top-level directory.

The scripts that use the Bernstein & Spicer model to generate the simulated A-scan sets used in the paper are [`bernspice_sdh.py`](https://github.com/MatthewRiding/Figures-laser-s-wave-DAS/blob/main/bernspice_sdh.py) and [`bernspice_hfs`](https://github.com/MatthewRiding/Figures-laser-s-wave-DAS/blob/main/bernspice_hfs.py) for the side-drilled hole (SDH) and horizontal plane reflector (HFS) experiments respectively.

# Beam profiling:

The digital camera images used for beam profiling of the generation laser beam are stored in the [`beam_profile_data`](https://github.com/MatthewRiding/Figures-laser-s-wave-DAS/tree/main/beam_profile_data) directory, alongside the MATLAB scripts used to fit Gaussian profiles to the camera data.  The results of beam profiling are summarised in [`beam profiling figure.jpg`](https://github.com/MatthewRiding/Figures-laser-s-wave-DAS/blob/main/beam_profile_data/Beam%20profiling%20figure.jpg).
