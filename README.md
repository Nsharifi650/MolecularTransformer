# MolecularTransformer
#### Molecular Structure Prediction using Transformers

This project involves predicting molecular structures based on given desired physical properties using a Transformer-based neural network model. The model takes physical properties such as polar surface area, molecular complexity, heavy atom count, hydrogen bond donors, and hydrogen bond acceptors as input and generates the corresponding molecular structure in the form of a SMILES (Simplified Molecular Input Line Entry System) string.

This repository is the source code for the article: [Generative AI: Transformers For Molecular Design](https://medium.com/ai-advances/transformers-for-molecular-generation-7434f5bef37a)

#### Project Motivation
The ability to predict molecular structures based on desired properties is of significant interest in fields such as drug discovery, material science, and chemistry. Traditional methods often require substantial computational resources and time. This project leverages the Transformer architecture, originally designed for natural language processing tasks, to generate valid molecular structures efficiently and accurately based on a set of input properties.

#### Data Description
The dataset used in this project includes molecules represented by their SMILES strings, along with the following physical properties:

Polar Surface Area (polararea): The sum of surfaces of polar atoms, usually related to solubility and permeability.
Molecular Complexity (complexity): A measure of the complexity of the molecule's structure.
Heavy Atom Count (heavycnt): The number of non-hydrogen atoms in the molecule.
Hydrogen Bond Donors (hbonddonor): The number of hydrogen atoms attached to electronegative atoms capable of forming hydrogen bonds.
Hydrogen Bond Acceptors (hbondacc): The number of electronegative atoms capable of accepting hydrogen bonds.
The SMILES strings serve as the target output for the model, and the physical properties are the input features.

#### Model Architecture

The model architecture is based on the Transformer model, with the following components:

Encoder: Takes the physical properties as input and encodes them into a latent space.
Decoder: Generates the SMILES string token by token, conditioned on the encoded properties.
Positional Encoding: Used in the decoder to provide information about the position of each token in the sequence.
The model was trained using a dataset of molecules with known SMILES strings and corresponding physical properties. During training, the model learns to generate valid SMILES strings that match the input properties.

#### Key Features:
Multi-Head Attention: Allows the model to focus on different parts of the input simultaneously.
Feed-Forward Neural Networks: Applied after attention layers to introduce non-linearity.
Positional Encoding: Injects sequential information into the model, crucial for generating valid SMILES strings.
Masking: Includes both padding and look-ahead masks to ensure correct sequence generation.
Installation
